"""
Liquid prompt formatter for different tasks.
"""

import random

LIQUID_TEXT_TO_IMAGE_INSTRUCTION_PROMPTS = [
    ' Generate an image based on this description.',
    ' Create an image that captures the provided description.',
    ' Based on the previous text, produce a corresponding image.',
    ' Please illustrate the above text with a picture.',
    ' Translate the given description into a image.',
    ' Construct a visual representation of the above description.',
    ' Create a image that matches the text.',
    ' Formulate a visual expression that reflects the narrative just provided.',
    ' Give a visual depiction based on the above sentences.',
    ' Create an image using the information mentioned above as guidance.',
]

LIQUID_IMAGE_CAPTIONING_INSTRUCTION_PROMPT = "The caption of this image is:"


class LiquidPromptFormatter:
    def __init__(self, task="text_to_image"):
        self.task = task

    def __call__(self, prompt, answer=None, mode="inference", apply_conversation_format: bool = False):
        if not isinstance(prompt, list):
            return self._apply(prompt=prompt, answer=answer, mode=mode, apply_conversation_format=apply_conversation_format)
        else:
            if answer is None:
                answer = [None] * len(prompt)
            return [
                self._apply(prompt=x, answer=y, mode=mode, apply_conversation_format=apply_conversation_format)
                for x, y in zip(prompt, answer)
            ]

    def _apply(self, prompt, answer=None, mode="inference", apply_conversation_format: bool = False):
        if self.task == "text_to_image":
            formatted_prompt = self.format_text_to_image_prompt(prompt=prompt, mode=mode)
        elif self.task == "image_to_text":
            formatted_prompt = self.format_image_to_text_prompt(prompt=prompt, answer=answer)
        elif self.task == "text_to_text":
            formatted_prompt = LIQUID_IMAGE_CAPTIONING_INSTRUCTION_PROMPT if prompt is None else prompt
        else:
            raise NotImplementedError

        return formatted_prompt

    @staticmethod
    def format_text_to_image_prompt(prompt, mode="inference"):
        if prompt == "<unconditional>":
            instruction = ""
        elif mode.lower() == "training":
            instruction = random.sample(LIQUID_TEXT_TO_IMAGE_INSTRUCTION_PROMPTS, k=1)[0]
        else:
            instruction = LIQUID_TEXT_TO_IMAGE_INSTRUCTION_PROMPTS[0]

        formatted_prompt = prompt + instruction + "<boi>"

        if mode.lower() == "training":
            formatted_prompt += "<eoi><eos>"

        return formatted_prompt

    @staticmethod
    def format_image_to_text_prompt(prompt=None, answer=None, mode="inference"):
        prompt = LIQUID_IMAGE_CAPTIONING_INSTRUCTION_PROMPT if prompt is None else prompt
        return "<boi><eoi>\n" + prompt

