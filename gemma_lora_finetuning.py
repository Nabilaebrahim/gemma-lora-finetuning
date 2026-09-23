# ============================================================
# Cell 1: Setup API keys as environment variables
# ============================================================
import os

os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
os.environ["KAGGLE_KEY"] = "your_kaggle_key"


# ============================================================
# Cell 2: Install required libraries
# ============================================================
# !pip install -q -U keras-nlp
# !pip install -q -U "keras>=3"


# ============================================================
# Cell 3: Select backend (Jax) and avoid memory fragmentation
# ============================================================
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import keras
import keras_nlp


# ============================================================
# Cell 4: Load and prepare the dataset (Databricks Dolly 15k)
# ============================================================
import json

data = []
with open("databricks-dolly-15k.jsonl") as file:
    for line in file:
        features = json.loads(line)
        # Skip examples that have extra context
        if features["context"]:
            continue
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(template.format(**features))

# Use only the first 1000 records for this demo
data = data[:1000]


# ============================================================
# Cell 5: Load the pre-trained Gemma model
# ============================================================
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()


# ============================================================
# Cell 6: Test the model BEFORE fine-tuning (baseline)
# ============================================================
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)

sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)

print(gemma_lm.generate(prompt, max_length=256))


# ============================================================
# Cell 7: Enable LoRA fine-tuning (rank = 4)
# ============================================================
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
# Note: trainable parameters drop from ~2.5B down to ~1.3M


# ============================================================
# Cell 8: Configure training (optimizer, loss, sequence length)
# ============================================================
gemma_lm.preprocessor.sequence_length = 512

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude bias and layer-norm scale from weight decay
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)


# ============================================================
# Cell 9: Run training
# ============================================================
gemma_lm.fit(data, epochs=1, batch_size=1)


# ============================================================
# Cell 10: Test the model AFTER fine-tuning
# ============================================================
prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)

print(gemma_lm.generate(prompt, max_length=256))