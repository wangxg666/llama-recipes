import copy

def is_standard_en_sent(sent: str):
    sent_copy = copy.deepcopy(sent)

    puncs = [
        ',', '.', ':', '?', '-', '_', '!', '@', '/', '\\',
        '[', ']', '{', '}', '(', ')', '=', '+', '|', '#', '$', '%',
        '^', '&', '\'', '"', '’'
    ]
    for punc in puncs:
        sent_copy = sent_copy.replace(punc, '')
    sent_copy = sent_copy.replace(' ', '')
    return sent_copy.encode('utf8').isalnum()


if __name__ == '__main__':
    for sent in [
        "'d you recognized the early signs of oral cancer?",
        "'ll Fargo Immaculate Endemic for Entire Financial Services Industry?",
        "'ll contribute in the smooth working of the kitchen and to assure that members receive the server standard of service they are promised.",
        "'ll the traditional path of learning to become a scholar official.",
        "'ll you know this would work with almond or paleo flour?",
        "'m ashamed of what happened in the White House yesterday,he said. ",
        "I think It 's a tragedy of the first proportion.",
        "'m sure there's several owners on here who will agree with my opinion (Xrac I believe is one).",
        "'re generating custom event recommendations for you based on Soca Dance Fitness right now!",
        "'re your postes getting less engygemenrt (less likes and comments)?",
        "'s 2 W -50 7.5 33 1-Janice Hazel-Rosemarie Laursen Namely vs.",
    ]:
        print(is_clean_sent(sent))

    sent = "이전 글이전 Inspiring Recomennaeions what the government act on Startup Nation!"
    print(is_clean_sent(sent))