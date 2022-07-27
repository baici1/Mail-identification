import argparse
import json
import random
import numpy as np
import torch
from flask import  Flask
import flask
from model import Bert

from pre_processing import pre_processing
app = Flask(__name__)

class Test:
    def __init__(self, config) -> None:
        self.pre_train_model = config["pre_train_model"]
        self.model_path = config["model_path"]
        self.max_length = config["max_length"]
        self.classific_num = config["classific_num"]
        self.device = config["device"]
        self.seed = config["seed"]
        self.resume = config["resume"]
        
    def setup_seed(self) -> None:
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
    
    def build_model(self) -> None:
        self.model = Bert(self.pre_train_model, self.classific_num, self.resume)
    
    def load_model(self) -> None:
        params = torch.load(self.model_path)
        self.model.load_state_dict(params["model"])
        
    def test(self, text) -> None:
        self.model.eval()
        mail_token = pre_processing(text, self.pre_train_model, self.max_length)
        input_ids = mail_token["input_ids"]
        attention_mask = mail_token["attention_mask"]
        token_type_ids = mail_token["token_type_ids"]
        with torch.no_grad():
            out = self.model([input_ids, attention_mask, token_type_ids])
            _, pred_label = out.max(1)
            confidence = torch.softmax(out,dim=1)[0][pred_label[0]]
        return {"state":"ok", "pred_label":int(pred_label[0]), "confidence":float(confidence)}
        
    
@app.route("/result", methods=['POST'])    
def start_flask():
    text = flask.request.form["text"]
    try:
        return test.test(text)
    except:
        return {"state":"fail", "information":"wrong format"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/test_config.json",
        help="the path of test config",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
        print("config:")
        print(json.dumps(config, indent=4))
    test = Test(config=config)
    test.setup_seed()
    test.build_model()
    test.load_model()
    app.run(debug=True)