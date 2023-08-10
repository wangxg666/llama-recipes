class HallucinationPrompt:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        prompt = """
            You are an excellent analyst, and I want you to analyze the following content, 
            and output the entities with their hypernym relationship or casuation from the content.
            The entities in your output must be **copied** from the content and no normalization is allowed.
            Please generated the output as followed json format, 
            ```json
            {
                'nodes': [
                    {
                        'entity': 'AAA',
                        'hyponym entity': 'BBB',
                        'hyponym relationship': 'CCC',
                        'casuation': 'DDD'
                    }
                ]
            }
            ```
            the `nodes` contains list of entities with 
            - AAA for entity name, 
            - BBB for hyponym entity name if eixst, 
            - CCC for hyponym relationship if exist, 
            - DDD for casuation of AAA,
            and both BBB, CCC, and DDD can be empty.
            Remember, all AAA, BBB, CCC, DDD should be copied from the content, and don't do any normalization.
            Here is the content: {content}
        """
        return prompt.replace("{content}", self.content)

