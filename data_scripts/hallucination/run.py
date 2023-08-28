from hallucination.prompt_util import HallucinationPrompt
from utils.openai_utils import *


if __name__ == '__main__':
    content = """
        Moyne Johnson, a resident at Good Samaritan Society – Loveland Village in Colorado, shares her experiences and advice on how to thrive in long-term care. Johnson has been a resident at Loveland Village for 11 years and is popular among the staff and residents, with senior living administrator Holly Turner stating that Johnson's contributions are "too much for me to comprehend". Johnson's activities include volunteering at the community's Country Store, playing cards with friends five times a week, serving as president of the resident council, and participating in three Bible studies.
        Johnson believes that the sense of community at Loveland Village is a key factor to thriving in long-term care. She explains that the facility's residents are allowed and encouraged to explore the area, participate in a variety of activities, and interact with other residents and staff. This is in line with Loveland Village's "culture of engagement" where residents are listened to, given a say, and feel a sense of ownership and home.
        Environmental services supervisor Christine Gibbs, who has become close to Johnson, describes her as "much more than a resident here" and "like my family". Johnson's positivity and involvement in the community has earned her the affection and respect of both the staff and her fellow residents.
        Even though Johnson had three back surgeries prior to moving into Loveland Village, she remains active, thanks to the help of the caregivers. She appreciates the assistance and care she receives and believes that it is "one of the best places in the world to live".
        In conclusion, Johnson's story provides a positive example of how to thrive in long-term care. Her active lifestyle, community involvement, and ability to form meaningful relationships with other residents and staff have helped her to live a fulfilling life in Loveland Village. Her experience underlines the importance of a supportive and engaging environment for residents to feel at home in long-term care facilities.
    """
    prompt = str(HallucinationPrompt(content))
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    if 'choices' not in response and len(response['choices']) == 0:
        exit(0)

    choice = response['choices'][0]
    print(choice)
