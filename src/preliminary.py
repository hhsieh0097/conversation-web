class Preliminary(object):
    def __init__(self, lang) -> None:
        if lang == 'en':
            self.init_info = "Terms and Conditions"
            self.init_info_first = "This system offers emotional support and assesses users for potential signs of depression. Results are for reference only and not a substitute for professional diagnosis. The system's evaluations are supplementary and not a replacement for expert assessment."
            self.init_info_second = "When you wish to end the conversation, please type \"conversation end\" (without the quotation marks). This will provide you with the results regarding your depressive tendencies."
            self.init_info_end = "Once you understand and accept the system, please press the Enter key in the input field to start the system."

            self.ask_info = "Before starting the system, please answer the following questions."

            self.portrait_info = "In order to predict the user's depressive tendencies more accurately, this system needs to collect your personal information. Please select your biological gender below and enter your age."

            self.ask_name = "Name: "
            self.ask_gender = "Gender: "

            self.start_info = "Start System"

            self.placeholder_info = "Write Something ..."

            self.warning_info = "Please confirm whether the age information is entered correctly."

            self.greeting = "Hello, Is there anything you want to talk to me about today?"

            self.depression_response = "According to the results of the evaluation of this system, your Depression Scale is at a high level. However, please note that this is for reference only and you are advised to seek professional help if necessary."
            self.nondepression_response = "According to the results of the evaluation of this system, your Depression Scale is at a low level. However, please note that this is for reference only and you are advised to seek professional help if necessary."
        elif lang == 'zh':
            self.init_info = "使用需知"
            self.init_info_first = "本系統的目標是為使用者提供情緒支援，並判斷其憂鬱的傾向，但其結果僅供參考。請注意，該系統的判斷結果僅作為輔助參考，並不具備專業診斷的功能。"
            self.init_info_second = "在希望結束聊天時請輸入 \"結束對話\" (不需輸入 \" 符號) 即可為您提供在憂鬱情緒的結果。"
            self.init_info_end = "了解並接受該系統之後，請在輸入欄中按下 Enter 鍵開始本系統。"

            self.ask_info = "開始本系統之前，請先回答下述幾個問題。"

            self.portrait_info = "為了能夠更精確地預測使用者的憂鬱傾向，本系統需要收集您的個人資料。請在下方選擇您的生理性別，以及輸入您的年齡資訊。"

            self.ask_name = "請問您的名字是？"
            self.ask_gender = "請問您的生理性別是？(男/女)"

            self.start_info = "開始系統"

            self.placeholder_info = "輸入 ... (當希望系統結束時請輸入：結束對話)"

            self.warning_info = "請確認年齡訊息是否正確輸入。"

            self.greeting = "你好，今天有什麼事情想跟我聊聊嗎？"

            self.depression_response = "根據本系統的評估結果，您的憂鬱傾向偏高。然而，請注意此結果參考之用，如有需要，建議您尋求專業醫師的協助。"
            self.nondepression_response = "根據本系統的評估結果，您的憂鬱傾向較低。儘管如此，請注意此結果僅供參考之用，若有需要，建議您尋求專業醫師的協助。"