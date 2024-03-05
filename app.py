from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,)

model_name = "ahmedabdo/arabic-summarizer-bart"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
app = Flask(__name__)
CORS(app)
@app.route('/summarize', methods = ['POST'])
def summarize():
    data = request.get_json()
    try:
        text = data['text']
        beams = data['beams']
        max_length = data['max_length']
        
        generated_output = model.generate( input_ids=tokenizer.encode(text, return_tensors="pt"),
        max_length=max_length or 128,  
        num_beams= beams or 4,
        early_stopping=True
        )
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        res = {"summary":generated_text}
        return jsonify(res),200
    except Exception as err:
        print(err)
        
    return "Something went wrong", 401
if(__name__=='__main__'):
    app.run(debug=True)