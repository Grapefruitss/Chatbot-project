from flask import Flask, request
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
from telegram import Update, Bot
from telegram.ext import CommandHandler, MessageHandler, Application, CallbackContext, filters
import asyncio

app = Flask(__name__)

TELEGRAM_TOKEN = '7453157582:AAFLidbVZ1RCR-WiDZO0X8WLmmbXCUpsV0U'
bot = Bot(token=TELEGRAM_TOKEN)

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id
max_length = 71
vocab_size = tokenizer.vocab_size
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=max_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size, max_len=max_length):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          batch_first=True)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer(src, tgt,
                                  tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)
    
device = torch.device('cpu')

def generate_text(model, src, tokenizer, max_length=max_length, device=device):
    src = src.to(device)
    src_key_padding_mask = (src == pad_token_id).to(device)

    with torch.no_grad():
        tgt_input = torch.tensor([[cls_token_id]], device=device, dtype=torch.long)
        for _ in range(max_length):
            tgt_key_padding_mask = (tgt_input == pad_token_id).to(device)

            output = model(src, tgt_input, src_key_padding_mask=src_key_padding_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask)
            output_probs = F.softmax(output[:, -1, :], dim=-1)
            next_token = torch.argmax(output_probs, dim=-1)

            tgt_input = torch.cat((tgt_input, next_token.unsqueeze(0)), dim=1)

            if next_token.item() == sep_token_id:
                break

    return tgt_input[:, 1:].squeeze(0)

def decode_output(output_tokens, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(output_tokens)
    return tokenizer.convert_tokens_to_string([token for token in tokens if token not in [tokenizer.pad_token, tokenizer.sep_token, tokenizer.unk_token]])

model = TransformerModel(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                          vocab_size=vocab_size, max_len=max_length)
model.to(device)
model.load_state_dict(torch.load('best_transformer_model.pth', map_location=device, weights_only=True))
model.eval()

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('서울시 맛집 추천 챗봇입니다 반가워요!')
    await update.message.reply_text('서울시 구별로 맛집을 추천드려요!')

async def respond(update: Update, context: CallbackContext):
    user_input = update.message.text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    outputs = generate_text(model, inputs['input_ids'], tokenizer, device=device)
    response = decode_output(outputs, tokenizer)
    await update.message.reply_text(response)
    if '맛집' in response:
        await update.message.reply_text('메뉴, 휴무일, 인접한 시설, 연락처, 위치, 주차시설, 영업시간 등을 물어보세요!')

application = Application.builder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler('start', start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

# 메인 실행 함수
if __name__ == '__main__':
    try:
        application.run_polling()
    except KeyboardInterrupt:
        print("Bot stopped manually")