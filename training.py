import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# 1. CONFIGURACIÓN INICIAL
MODEL_NAME = "distilbert/distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Usará tu CPU Intel de forma segura
print(f"🖥️ Ejecutando el entrenamiento en: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 no tiene token de relleno por defecto, usamos el de fin de texto
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# 2. PREPARACIÓN DE LOS DATOS (Dataset personalizado)
class DatasetComputadoras(Dataset):
    def __init__(self, txt_path, tokenizer, max_length=64):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.read().split("<|endoftext|>")
        
        self.input_ids = []
        self.attention_masks = []
        
        for line in lines:
            if line.strip():
                # Forzamos a que el texto empiece con el token de inicio
                texto = tokenizer.eos_token + line
                encodings = tokenizer(texto, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
                self.input_ids.append(encodings["input_ids"].squeeze(0))
                self.attention_masks.append(encodings["attention_mask"].squeeze(0))
                
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx]

# Cargamos los datos en lotes pequeños (Batch Size = 1) para no saturar tu RAM
dataset = DatasetComputadoras("data.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 3. OPTIMIZADOR
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. BUCLE DE ENTRENAMIENTO PROPIO
model.train()
print("\n🚀 Iniciando Fine-Tuning. Tu modelo esta estudiando el archivo...")

for epoch in range(15): # 15 vueltas al archivo de texto seran suficientes
    total_loss = 0
    for input_ids, attention_mask in dataloader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # En GPT, las "labels" (respuestas correctas) son los mismos inputs movidos un lugar a la derecha.
        # Hugging Face hace este desplazamiento de forma interna si le pasamos labels=input_ids
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()  # Limpiar gradientes anteriores
        loss.backward()        # Backpropagation (Calcular error de cada neurona)
        optimizer.step()       # Ajustar pesos moleculares de la IA
        
        total_loss += loss.item()
        
    print(f"📉 Época [{epoch+1}/15] -> Pérdida (Error): {total_loss/len(dataloader):.4f}")

# 5. GUARDAR TU NUEVO MODELO MODIFICADO
print("\n💾 ¡Entrenamiento terminado! Guardando el nuevo cerebro localmente...")
model.save_pretrained("./mi_ia_computadoras")
tokenizer.save_pretrained("./mi_ia_computadoras")
print("🎉 Listo. Carpeta './mi_ia_computadoras' creada con éxito.")
