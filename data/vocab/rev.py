import csv

# 입력 .vocab 파일 경로와 출력 .csv 파일 경로
vocab_file_path = '/Users/gangjiyeon/Downloads/demo/kospeech/data/vocab/kspon_sentencepiece.vocab'
csv_file_path = '/Users/gangjiyeon/Downloads/demo/kospeech/data/vocab/kspon_sentencepiece.csv'

# 특수 토큰 정의
special_tokens = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    '<unk>': 3,
    '<blank>': 4
}

# .vocab 파일 읽기
with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
    vocab_lines = vocab_file.readlines()

# .csv 파일 쓰기
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    # 헤더 작성
    csv_writer.writerow(['id', 'character'])

    # 특수 토큰 먼저 작성
    for token, id_val in special_tokens.items():
        csv_writer.writerow([id_val, token])

    # 각 단어와 ID 처리
    for line in vocab_lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            character, id_str = parts
            # 음수 ID 값을 양수로 변환
            id_val = abs(int(id_str))
            csv_writer.writerow([id_val, character])

print(f"{csv_file_path} 파일로 변환되었습니다.")
