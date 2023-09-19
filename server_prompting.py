import json
import logging
import datetime
from deep_translator import GoogleTranslator
from flask import Flask, render_template, request

from src.config import parse_args
from src.preliminary import Preliminary
from src.emo_sup.prompting.chat import EmoSupModel
from src.depression.personal_detect import PersonalDetection

logging.getLogger('transformers').setLevel(logging.ERROR) # To avoid the warning of transformers package. 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template(
        "index.html",
        preliminary_info=_Preliminary.init_info, 
        preliminary_info_first=_Preliminary.init_info_first,
        preliminary_info_second=_Preliminary.init_info_second,
        preliminary_info_end=_Preliminary.init_info_end, 
        portrait_info=_Preliminary.portrait_info, 
        start_info=_Preliminary.start_info,
        warning_info=_Preliminary.warning_info, 
        greeting_info=_Preliminary.greeting,
        placeholder_text=_Preliminary.placeholder_info,
    )


@app.route("/save_porfile", methods=['POST'])
def save_porfile():
    data = request.get_json()

    global gender, age
    age, gender = data.get('age'), data.get('gender')
    
    if gender == 'radio-male': gender = 'Male'
    elif gender == 'radio-female': gender = 'Female'

    print('[Profile] - Age: {}, Gender: {}'.format(age, gender))

    response = {'message': True}

    return response


@app.route("/get_method", methods=["GET"])
def get_method():
    if request.method == "GET":
        return "GET"
    else:
        return render_template("index.html")


@app.route("/post_method", methods=["POST"])
def post_method():
    if request.method == "POST":
        context = json.loads(request.form.get("data"))
        context = [{k: v.splitlines()[0]} for cont in context for k, v in cont.items()]
        # print('[Context]: ', context)

        if '結束對話' in context[-1]['User'] or 'conversation end' in context[-1]['User']:
            last = True
            
            user_transcripts = list()
            for utter in context:
                if 'User' in utter.keys(): user_transcripts.append(utter['User'])

            if args.lang == 'zh':
                user_transcripts = GoogleTranslator(source='zh-TW', target='en').translate_batch(user_transcripts)

            phq_score, depression = _DeprDetection.get_depression(user_transcripts[: -1], gender)
            print('[PHQ-Score]: {}'.format(phq_score))
            
            if depression == 1: 
                response = _Preliminary.depression_response
            else:
                response = _Preliminary.nondepression_response

            save_file(age, gender, phq_score, context)
        else:
            last = False
            response, cot_response = _EmoSupModel.chatting(context=context)

            print('[Chain-of-Thought]: ')
            for output_step in cot_response:
                print(output_step)

        return json.dumps({"response": response, "last": last})
    else:
        return render_template("index.html")


def save_file(age, gender, phq_score, context):
    now = datetime.datetime.now()
    time = str(now.month) + str(now.day) + '_' + now.strftime('%H%M')

    with open(f'./storage/{time}.json', 'w', encoding='utf-8') as f:
        json.dump({'age': age, 'gender': gender, 'phq': phq_score, 'context': context}, f, ensure_ascii=False)

    return


if __name__ == "__main__":
    args = parse_args()

    _Preliminary = Preliminary(lang=args.lang)
    _EmoSupModel = EmoSupModel(lang=args.lang)
    _DeprDetection = PersonalDetection(vad_trained_path=args.vad_trained_path, detec_trained_path=args.detec_trained_path)

    app.run(debug=True, host="192.168.50.217", port=57150)