import re

special_chars_remover = re.compile("[^\w'|_]")

def main():
    sentence = "Bag-of-Words 모델을 Python으로 직접 구현하겠습니다."
    bow = create_BOW(sentence)
    
    print(bow)


def create_BOW(sentence):
    bow = {}
    sentence_lowered = sentence.lower()
    sentence_without_s_c = remove_special_characters(sentence_lowered)
    splitted_sentence = sentence_without_s_c.split()
    
    splitted_sentence_filtered = [
                                  token
                                  for token in splitted_sentence
                                  if len(token) >= 1
                                  ]
        
                                  for token in splitted_sentence_filtered:
                                      # if token not in bow:
                                      #     bow[token] = 1
                                      # else:
                                      #     bow[token] += 1
                                      # 아래랑 같은 코드
                                      bow.setdefault(token,0) #만약 token이 없으면 0으로 정해라
                                      bow[token]+=1 # 그리고 거기다가 1을 더해줘라
                                  return bow

def remove_special_characters(sentence):
    return special_chars_remover.sub(' ', sentence)

if __name__ == "__main__":
    main()
