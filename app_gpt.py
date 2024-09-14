import openai

# Khóa API của OpenAI, thay bằng khóa của bạn
openai.api_key = "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"

def ask_gpt(question):
    try:
        # Gọi API OpenAI để trả lời theo phiên bản mới
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Hoặc bạn có thể dùng model khác như gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        
        # Lấy câu trả lời từ API
        answer = response['choices'][0]['message']['content']
        
        # In câu trả lời ra console
        print("Câu trả lời từ GPT-4: ", answer)
    
    except Exception as e:
        print("Đã xảy ra lỗi:", str(e))

if __name__ == '__main__':
    while True:
        # Nhập câu hỏi từ người dùng
        question = input("Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát): ")
        if question.lower() == 'exit':
            break
        ask_gpt(question)
