import re

import os
import openai

from src.config import parse_args
args = parse_args()

os.environ['OPENAI_API_KEY'] = args.openai_key
openai.api_key = args.openai_key

class EmoSupModel(object):
    def __init__(self, lang) -> None:
        if lang == 'zh':
            self.system_prompt = "你將扮演 Assistant 的角色，並且滿足下述的所有描述。" + '\n\n' + \
                                 "Assistant 在對話初期時會著重詢問 User 的感受、情況以及狀態。對話中期則是在確認 User 面臨的問題之後表達同情、鼓勵，或是與 User 建立共鳴讓 User 得到安慰以及被理解。對話後期則是以提供協助或建議為主幫助 User 解決問題。" + '\n\n' + \
                                 "Assistant 會先說明 User 現在的問題和情況，並且解釋其引發的原因。然後判斷過去 Assistant 做過的事情有哪些，接著找出 User 最後一次的回覆是什麼，並且從 Support Strategies 中選擇一個 Support Strategy，然後解釋選擇的原因。最後透過這個 Strategy 生成一個相對應的回覆給 User。" + '\n\n' + \
                                 "Support Strategies: [Question], [Restatement or Paraphrasing], [Reflection of feelings], [Self-disclosure], [Affirmation and Reassurance], [Providing Suggestions], [Information], [Others]" + '\n\n' + \
                                 "Support Strategies 定義：" + '\n' + \
                                 "[Question]: 當 Assistant 需要詢問問題來幫助 User 知道他們正在面對什麼問題時會使用這個 Strategy。" + '\n' + \
                                 "[Restatement or Paraphrasing]: Assistant 透過這個 Strategy 重述 User 的情況來幫助 User 知道他現在的情況。" + '\n' + \
                                 "[Reflection of feelings]: 當需要闡明或描述 User 的感受時，Assistant 可以使用這個 Strategy。" + '\n' + \
                                 "[Self-disclosure]: 當需要與 User 建立共鳴或分享具有經驗性質的回覆時，Assistant 可以使用這個 Strategy。" + '\n' + \
                                 "[Affirmation and Reassurance]: 當需要肯定 User 的能力或給予鼓勵和安慰時，Assistant 可以使用這個 Strategy。" + '\n' + \
                                 "[Providing Suggestions]: 當需要提供改變的建議時，Assistant 會使用這個 Strategy 來提供一些建議或解決方案。" + '\n' + \
                                 "[Information]: 當需要提供有關特定主題的知識或資訊時，Assistant 可以使用這個 Strategy。" + '\n' + \
                                 "[Others]: 當以上策略都不適用或是要表達噓寒問暖的問候時，Assistant 可以使用這個策略。" + '\n\n' 
            self.conversation_prompt = "Conversation History: " + '\n'
            self.last_message_prompt = '\n\n' + '=== User 最後的語句 ===' + '\n'
            self.seperate_prompt = '\n\n###\n\n'
            self.instruct_prompt = "Assistant 會根據事實回答 Demand Format 的問題，並且會以 Demand Format 的形式進行回覆。" + '\n\n' + \
                                   "Demand Format:" + '\n' + \
                                   "1. User 現在的情況和問題：" + '\n' + \
                                   "2. User 是否說明該情況和問題的原因：" + '\n' + \
                                   "3. Assistant 上次做了什麼事情：" + '\n' + \
                                   "4. === User 最後的語句 === 的完整句子是什麼：" + '\n' + \
                                   "5. 根據 === User 最後的語句 === 選擇 Support Strategy：" + '\n' + \
                                   "6. 選擇該 Support Strategy 的原因：" + '\n' + \
                                   "7. 回覆 === User 最後的語句 === (最多 5 句話)："
        elif lang == 'en':
            self.system_prompt = "You will play the role of Assistant and satisfy all of the descriptions below." + '\n\n' + \
                                 "The Assistant will focus on asking the User about the feelings, situation and status at the first stage of the conversation. During the middle stage of the conversation, the Assistant will identify the problems the User is facing and then express sympathy, encouragement or empathy with the User so that the User is comforted. In the end stage of the conversation, the main focus is to provide assistance or suggestions to help the User solve the problem." + '\n\n' \
                                 "The Assistant first describes the User's current problem and situation and explains the reasons for it. It then determines what the Assistant has done in the past, finds out the User's last utterance, selects a Support Strategy from the Support Strategies and explains why it was chosen. In the end, an appropriate response is given to the User." + '\n\n' \
                                 "Support Strategies: [Question], [Restatement or Paraphrasing], [Reflection of feelings], [Self-disclosure], [Affirmation and Reassurance], [Providing Suggestions], [Information], [Others]" + '\n\n' + \
                                 "Support Strategies Definitions:" + '\n' \
                                 "[Question]: This Strategy is used when the Assistant needs to ask a question to help the User know what problem it is facing." + '\n' \
                                 "[Restatement or Paraphrasing]: Assistant can use this Strategy to restate the User's situation to help the User know what they are facing." + '\n' \
                                 "[Reflection of feelings]: Assistant can use this Strategy when it needs to clarify or describe User's feelings." + '\n' \
                                 "[Self-disclosure]: This Strategy can be used by the Assistant when there is a need to empathize with the User or share an experience based reply." + '\n' \
                                 "[Affirmation and Reassurance]: This Strategy can be used by the Assistant when there is a need to affirm the User's abilities or provide encouragement and reassurance." + '\n' \
                                 "[Providing Suggestions]: When there is a need to provide suggestions for change, the Assistant will use this Strategy to provide some suggestions or solutions." + '\n' \
                                 "[Information]: This Strategy can be used by Assistant when there is a need to provide knowledge or information about a specific topic." + '\n' \
                                 "[Others]: This Strategy is used when none of the above strategies are appropriate, or when the Assistant wants to offer a warm and friendly greeting." + '\n\n'
            self.conversation_prompt = "Conversation History: " + '\n'
            self.last_message_prompt = '\n\n' + "=== User's last utterance ===" + '\n'
            self.seperate_prompt = '\n\n###\n\n'
            self.instruct_prompt = "Assistant will answer Demand Format questions based on facts and will respond in Demand Format." + '\n' \
                                   "Demand Format: " + '\n' \
                                   "1. User's current situation and problem: " + '\n' \
                                   "2. The reason for causing the User's current situation and problem: " + '\n' \
                                   "3. What the Assistant has done: " + '\n' \
                                   "4. What is the full statement in the === User's last utterance ===: " + '\n' \
                                   "5. Select Support Strategy based on === User's last statement ===: " + '\n' \
                                   "6. The reason for choosing the Support Strategy: " + '\n' + \
                                   "7. Reply === User's last sentence === (up to 5 sentences): "
        
        self.generate_kwargs = {
            'temperature': args.temparature, 
            'top_p': args.top_p, 
            'frequency_penalty': args.frequency_penalty, 
            'presence_penalty': args.presence_penalty
        }
    
    def preprocess_conversation(self, context, system_prompt, max_turn):
        _context = list()
        for idx in range(len(context)):
            for k, v in context[idx].items():
                _context.append('{}：{}'.format(k, v))

        if max_turn == -1: pass
        else: _context = _context[-(max_turn * 2): ]
        
        print('[Context]: \n' + '\n'.join(_context))
        
        _context[-1] = _context[-1] + self.last_message_prompt + _context[-1]
        _context = system_prompt + self.conversation_prompt + '\n'.join(_context) + self.seperate_prompt + self.instruct_prompt
        
        full_prompt = [{'role': 'system', 'content': _context}]

        return full_prompt

    def postprocess_response(self, response):
        top_response = response['choices'][0]['message']['content']

        pattern = r'\b\d+\..+?(?=\n\d+\.|\Z)'
        cot_response = re.findall(pattern, top_response, re.DOTALL)
        if len(cot_response) > 7:
            cot_response[6] = '\n'.join(cot_response[6: ])
            cot_response = cot_response[: 7]
    
        assistant_response = cot_response[-1]

        if ':' in assistant_response or '：' in assistant_response:
            assistant_response = re.split(r':|：', assistant_response, 1)[-1].strip()

        if assistant_response.startswith('7. '): 
            assistant_response = assistant_response[3: ].strip()

        if assistant_response.startswith('回覆'):
            assistant_response = assistant_response[2: ].strip()

        if 'Assistant：' in assistant_response:
            assistant_response = re.sub('Assistant：', '', assistant_response)

        if assistant_response[0] == '\"' or assistant_response[0] == '「' or assistant_response[0] == '[':
            assistant_response = assistant_response[1: ]

        if assistant_response[-1] == '\"' or assistant_response[-1] == '」' or assistant_response[-1] == ']':
            assistant_response = assistant_response[: -1]

        if assistant_response.startswith('7. '): assistant_response = assistant_response[3: ]
        if assistant_response[0] == '\"' or assistant_response[0] == '「' or assistant_response[0] == '[':
            assistant_response = assistant_response[1: ]
        if assistant_response[-1] == '\"' or assistant_response[-1] == '」' or assistant_response[-1] == ']':
            assistant_response = assistant_response[: -1]

        return assistant_response, cot_response
    
    def chatting(self, context):
        processed_context = self.preprocess_conversation(context, self.system_prompt, args.max_turn)

        flag = False
        while flag == False:
            try:
                response = openai.ChatCompletion.create(
                    model=args.model_name, 
                    messages=processed_context, 
                    **self.generate_kwargs
                )

                flag = True
            except openai.OpenAIError as e:
                flag = False

        assistant_response, cot_response = self.postprocess_response(response)

        return assistant_response, cot_response