# -*- coding: utf-8 -*-
import os
import gc
import random
import numpy as np
import pandas as pd
import torch

from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
from transformers.optimization import Adafactor


# ============================================================
# 0. 基本設定
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-large"

TRAIN_PATH = "train_triple_augmented.csv"
TEST_PATH = "test_no_answer_2022.csv"

TEXT_COL = "TEXT"
LABEL_COL = "LABEL"
ROW_ID_COL = "row_id"

N_SPLITS = 5
SEED = 42

MAX_LENGTH = 256
NUM_EPOCHS = 3

# pseudo label 設定：0/1 各取前 5%
PSEUDO_TOP_FRAC_EACH_CLASS = 0.05

OUTPUT_DIR = "./deberta_large_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 固定隨機種子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ============================================================
# 2. NaN 防護 callback
# ============================================================

class EarlyNaNCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        if state.global_step < 50:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"\n!!! 訓練崩潰：step {state.global_step}，參數 {name} 出現 NaN/Inf")
                        control.should_training_stop = True
                        raise ValueError("偵測到 NaN/Inf 梯度，停止訓練。")


# ============================================================
# 3. metrics：用 macro_f1 選最佳模型
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_neg": f1_score(labels, preds, pos_label=0),
        "f1_pos": f1_score(labels, preds, pos_label=1),
    }


# ============================================================
# 4. threshold search
# ============================================================

def search_best_threshold(oof_probs, y_true):
    results = []

    for th in np.arange(0.35, 0.751, 0.005):
        preds = (oof_probs[:, 1] >= th).astype(int)

        acc = accuracy_score(y_true, preds)
        macro = f1_score(y_true, preds, average="macro")
        f1_neg = f1_score(y_true, preds, pos_label=0)
        f1_pos = f1_score(y_true, preds, pos_label=1)

        results.append({
            "threshold": th,
            "accuracy": acc,
            "macro_f1": macro,
            "f1_neg": f1_neg,
            "f1_pos": f1_pos,
            "score": (acc + macro) / 2,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df.iloc[0], df


# ============================================================
# 5. 資料前處理
# ============================================================

def clean_dataframe(df, has_label=True):
    df = df.copy()

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()

    if has_label:
        df = df.dropna(subset=[TEXT_COL, LABEL_COL])
        df[LABEL_COL] = df[LABEL_COL].astype(int)
    else:
        df = df.dropna(subset=[TEXT_COL])

    return df.reset_index(drop=True)


# ============================================================
# 6. 5-fold 訓練主函式
# ============================================================

def run_kfold(train_df, test_df, stage_name="stage1", seed=42):
    print(f"\n========== {stage_name}: 開始 {N_SPLITS}-Fold 訓練 ==========")

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess(examples):
        return tokenizer(
            examples[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH
        )

    test_ds = Dataset.from_pandas(test_df[[TEXT_COL]], preserve_index=False)
    test_ds = test_ds.map(preprocess, batched=True)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    y = train_df[LABEL_COL].values

    oof_logits = np.zeros((len(train_df), 2), dtype=np.float32)
    test_logits_all_folds = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        fold_id = fold + 1
        print(f"\n========== {stage_name} Fold {fold_id}/{N_SPLITS} ==========")

        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        train_ds = Dataset.from_pandas(
            fold_train[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"}),
            preserve_index=False
        ).map(preprocess, batched=True)

        val_ds = Dataset.from_pandas(
            fold_val[[TEXT_COL, LABEL_COL]].rename(columns={LABEL_COL: "labels"}),
            preserve_index=False
        ).map(preprocess, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2
        )

        model.gradient_checkpointing_enable()

        optimizer = Adafactor(
            model.parameters(),
            lr=2e-5,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.01,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        fold_output_dir = os.path.join(OUTPUT_DIR, f"{stage_name}_fold{fold_id}")

        args = TrainingArguments(
            output_dir=fold_output_dir,

            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,

            num_train_epochs=NUM_EPOCHS,
            max_grad_norm=0.1,

            fp16=False,
            bf16=False,

            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,

            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,

            save_total_limit=1,
            report_to="none",

            gradient_checkpointing=True,
            torch_compile=False,

            seed=seed,
            data_seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
            callbacks=[EarlyNaNCheckCallback()]
        )

        trainer.train()

        val_pred = trainer.predict(val_ds)
        oof_logits[val_idx] = val_pred.predictions

        val_probs = softmax(val_pred.predictions, axis=1)
        val_preds = np.argmax(val_probs, axis=1)

        fold_acc = accuracy_score(fold_val[LABEL_COL].values, val_preds)
        fold_macro = f1_score(fold_val[LABEL_COL].values, val_preds, average="macro")

        fold_metrics.append({
            "stage": stage_name,
            "fold": fold_id,
            "accuracy": fold_acc,
            "macro_f1": fold_macro,
        })

        print(f"[{stage_name} Fold {fold_id}] accuracy={fold_acc:.5f}, macro_f1={fold_macro:.5f}")

        test_pred = trainer.predict(test_ds)
        test_logits_all_folds.append(test_pred.predictions)

        del model, trainer, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    test_logits = np.mean(test_logits_all_folds, axis=0)

    oof_probs = softmax(oof_logits, axis=1)
    test_probs = softmax(test_logits, axis=1)

    fold_metrics_df = pd.DataFrame(fold_metrics)

    return oof_logits, oof_probs, test_logits, test_probs, fold_metrics_df


# ============================================================
# 7. 建立類別平衡 pseudo label
# ============================================================

def build_balanced_pseudo_labels(test_df, test_probs, top_frac_each_class=0.05):
    pred = np.argmax(test_probs, axis=1)
    conf = np.max(test_probs, axis=1)

    idx0 = np.where(pred == 0)[0]
    idx1 = np.where(pred == 1)[0]

    n_each = int(len(test_df) * top_frac_each_class)

    n0 = min(n_each, len(idx0))
    n1 = min(n_each, len(idx1))

    idx0_top = idx0[np.argsort(conf[idx0])[-n0:]]
    idx1_top = idx1[np.argsort(conf[idx1])[-n1:]]

    pseudo_idx = np.concatenate([idx0_top, idx1_top])
    pseudo_idx = pseudo_idx[np.argsort(-conf[pseudo_idx])]

    pseudo_df = pd.DataFrame({
        TEXT_COL: test_df.loc[pseudo_idx, TEXT_COL].values,
        LABEL_COL: pred[pseudo_idx],
        "pseudo_confidence": conf[pseudo_idx],
        "pseudo_source": "deberta_large_stage1"
    })

    print("\n========== Pseudo Label 統計 ==========")
    print(pseudo_df[LABEL_COL].value_counts())
    print(pseudo_df["pseudo_confidence"].describe())

    return pseudo_df


# ============================================================
# 8. 儲存 submission
# ============================================================

def save_submission(test_df, test_probs, threshold, filename):
    preds = (test_probs[:, 1] >= threshold).astype(int)

    sub = pd.DataFrame({
        ROW_ID_COL: test_df[ROW_ID_COL],
        LABEL_COL: preds
    })

    path = os.path.join(OUTPUT_DIR, filename)
    sub.to_csv(path, index=False)

    print(f"Saved submission: {path}")
    print("Prediction distribution:")
    print(sub[LABEL_COL].value_counts())

    return sub


def save_probs(test_df, probs, filename):
    df = pd.DataFrame({
        ROW_ID_COL: test_df[ROW_ID_COL],
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "pred_05": np.argmax(probs, axis=1),
    })

    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)

    print(f"Saved probs: {path}")
    return df


# ============================================================
# 9. 主流程
# ============================================================

def run_workflow():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df = clean_dataframe(train_df, has_label=True)
    test_df = clean_dataframe(test_df, has_label=False)

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("\nTrain label distribution:")
    print(train_df[LABEL_COL].value_counts())

    # ------------------------------------------------------------
    # Stage 1：原始資料 5-fold
    # ------------------------------------------------------------

    oof_logits_1, oof_probs_1, test_logits_1, test_probs_1, fold_metrics_1 = run_kfold(
        train_df=train_df,
        test_df=test_df,
        stage_name="stage1",
        seed=SEED
    )

    np.save(os.path.join(OUTPUT_DIR, "stage1_oof_logits.npy"), oof_logits_1)
    np.save(os.path.join(OUTPUT_DIR, "stage1_test_logits.npy"), test_logits_1)

    save_probs(test_df, test_probs_1, "stage1_test_probs.csv")

    oof_df_1 = pd.DataFrame({
        "oof_prob_0": oof_probs_1[:, 0],
        "oof_prob_1": oof_probs_1[:, 1],
        LABEL_COL: train_df[LABEL_COL].values
    })
    oof_df_1.to_csv(os.path.join(OUTPUT_DIR, "stage1_oof_probs.csv"), index=False)

    best_1, threshold_df_1 = search_best_threshold(
        oof_probs=oof_probs_1,
        y_true=train_df[LABEL_COL].values
    )

    threshold_df_1.to_csv(os.path.join(OUTPUT_DIR, "stage1_threshold_search.csv"), index=False)
    fold_metrics_1.to_csv(os.path.join(OUTPUT_DIR, "stage1_fold_metrics.csv"), index=False)

    print("\n========== Stage 1 Best Threshold ==========")
    print(best_1)

    stage1_best_th = float(best_1["threshold"])

    save_submission(
        test_df=test_df,
        test_probs=test_probs_1,
        threshold=stage1_best_th,
        filename="submission_stage1_best_threshold.csv"
    )

    save_submission(
        test_df=test_df,
        test_probs=test_probs_1,
        threshold=0.5,
        filename="submission_stage1_threshold_050.csv"
    )

    # ------------------------------------------------------------
    # Stage 2：建立 pseudo label
    # ------------------------------------------------------------

    pseudo_df = build_balanced_pseudo_labels(
        test_df=test_df,
        test_probs=test_probs_1,
        top_frac_each_class=PSEUDO_TOP_FRAC_EACH_CLASS
    )

    pseudo_df.to_csv(os.path.join(OUTPUT_DIR, "pseudo_labels_balanced.csv"), index=False)

    final_train = pd.concat([
        train_df[[TEXT_COL, LABEL_COL]],
        pseudo_df[[TEXT_COL, LABEL_COL]]
    ], ignore_index=True)

    final_train = final_train.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    print("\n========== Final Train after Pseudo ==========")
    print(final_train.shape)
    print(final_train[LABEL_COL].value_counts())

    # ------------------------------------------------------------
    # Stage 3：合併 pseudo 後重訓
    # ------------------------------------------------------------

    oof_logits_2, oof_probs_2, test_logits_2, test_probs_2, fold_metrics_2 = run_kfold(
        train_df=final_train,
        test_df=test_df,
        stage_name="stage2_pseudo",
        seed=SEED
    )

    np.save(os.path.join(OUTPUT_DIR, "stage2_oof_logits.npy"), oof_logits_2)
    np.save(os.path.join(OUTPUT_DIR, "stage2_test_logits.npy"), test_logits_2)

    save_probs(test_df, test_probs_2, "stage2_test_probs.csv")

    oof_df_2 = pd.DataFrame({
        "oof_prob_0": oof_probs_2[:, 0],
        "oof_prob_1": oof_probs_2[:, 1],
        LABEL_COL: final_train[LABEL_COL].values
    })
    oof_df_2.to_csv(os.path.join(OUTPUT_DIR, "stage2_oof_probs.csv"), index=False)

    best_2, threshold_df_2 = search_best_threshold(
        oof_probs=oof_probs_2,
        y_true=final_train[LABEL_COL].values
    )

    threshold_df_2.to_csv(os.path.join(OUTPUT_DIR, "stage2_threshold_search.csv"), index=False)
    fold_metrics_2.to_csv(os.path.join(OUTPUT_DIR, "stage2_fold_metrics.csv"), index=False)

    print("\n========== Stage 2 Best Threshold ==========")
    print(best_2)

    stage2_best_th = float(best_2["threshold"])

    save_submission(
        test_df=test_df,
        test_probs=test_probs_2,
        threshold=stage2_best_th,
        filename="submission_stage2_pseudo_best_threshold.csv"
    )

    save_submission(
        test_df=test_df,
        test_probs=test_probs_2,
        threshold=0.5,
        filename="submission_stage2_pseudo_threshold_050.csv"
    )

    # ------------------------------------------------------------
    # 額外輸出：Stage1 + Stage2 同模型平均
    # ------------------------------------------------------------

    avg_probs = (test_probs_1 + test_probs_2) / 2

    save_probs(test_df, avg_probs, "stage1_stage2_avg_test_probs.csv")

    save_submission(
        test_df=test_df,
        test_probs=avg_probs,
        threshold=stage1_best_th,
        filename="submission_avg_stage1_threshold.csv"
    )

    save_submission(
        test_df=test_df,
        test_probs=avg_probs,
        threshold=stage2_best_th,
        filename="submission_avg_stage2_threshold.csv"
    )

    save_submission(
        test_df=test_df,
        test_probs=avg_probs,
        threshold=0.5,
        filename="submission_avg_threshold_050.csv"
    )

    # ------------------------------------------------------------
    # 總結
    # ------------------------------------------------------------

    summary = pd.DataFrame([
        {
            "version": "stage1",
            "best_threshold": stage1_best_th,
            "best_accuracy": best_1["accuracy"],
            "best_macro_f1": best_1["macro_f1"],
            "best_score": best_1["score"],
        },
        {
            "version": "stage2_pseudo",
            "best_threshold": stage2_best_th,
            "best_accuracy": best_2["accuracy"],
            "best_macro_f1": best_2["macro_f1"],
            "best_score": best_2["score"],
        }
    ])

    summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    print("\n========== 完成 ==========")
    print(summary)
    print(f"\n所有檔案已輸出到：{OUTPUT_DIR}")


if __name__ == "__main__":
    run_workflow()