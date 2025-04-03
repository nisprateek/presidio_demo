import logging
from typing import Optional, List, Tuple, Set
import re
import json

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation
)
from presidio_analyzer.nlp_engine import NlpArtifacts
from src.connectors.openai_connector import predict_with_gpt_4

class LlmRecognizer(EntityRecognizer):
    ENTITIES = [
        "LOCATION",
        "PERSON",
        "ORGANIZATION",
    ]

    DEFAULT_EXPLANATION = "Identified as {} by LLM Named Entity Recognition."

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"ORGANIZATION"}, {"ORG", "ORGANIZATION"}),
    ]

    PRESIDIO_EQUIVALENCES = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
    }

    def __init__(self, model_name: str, supported_entities: Optional[List[str]] = None,
                check_label_groups: Optional[Tuple[Set, Set]] = None):
        self.model_name = model_name
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )
        self.supported_entities = supported_entities if supported_entities else self.ENTITIES

        super().__init__(
            supported_entities=self.supported_entities,
            name=self.model_name
            
        )

    def get_supported_entities(self) -> List[str]:
        return self.supported_entities
    
    def load_model(self):
        pass

    def analyze(
            self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        Analyze the text and return a list of AnalysisExplanation objects.
        """
        if not entities:
            entities = self.supported_entities

        prompt = (
            "Given below is a text input that might contain various types of Personal Identifiable Information (PII).\n"
            "Your task is to extract  PII entities of the following types **ONLY**:\n"
            f"Entities: {', '.join(entities)}\n"
        
        "For each detected entity, output an object with the following keys:\n"
        "- 'entity_type': a string representing the type of PUU (e.g. PERSON, LOCATION, EMAIL)\n"
        "- 'entity_text': a string representng the verbatim PII text, exactly as it appears in the input\n"
        "- 'score': a float between 0 and 1, representing the confidence in the detection\n"
        "- 'explanation': a string explaining the reasoning behind the detection\n"

        "Make sure your output is **ONLY** a valid JSON array contaitning these objects, and nothing else. No text before or after the JSON array. Just a valid JSON array."
            
        "Now, extract the PII entities from the following text:\n"
        f"Text: {text}"
        )
        if "gpt_4o" in self.model_name:
            response_str = predict_with_gpt_4(prompt)
        else:
            raise ValueError("Invalid Model Name")
        
        print (f"RESPONSE STR from {self.model_name}: {response_str}")

        response_list = self._extract_json_from_response(response_str)

        recognizer_results = []
        for entry in response_list:
            start, end = self._find_text_indices(text, entry['entity_text'])
            entity_type = entry['entity_type']
            score = entry['score']
            analysis_explanation = AnalysisExplanation(
                recognizer=self.__class__.__name__,
                original_score=score,
                textual_explanation=entry['explanation']
            )
            recognizer_result = RecognizerResult(
                entity_type=entity_type,
                start=start,
                end=end,
                score=score,
                analysis_explanation=analysis_explanation
            )
            recognizer_results.append(recognizer_result)
        
        return recognizer_results
    
    def _extract_json_from_response(self, response:str):
        match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if match:
            json_content = match.group(0)
            return json.loads(json_content)
        else:
            raise ValueError("No JSON array found in the response")
    
    def _find_text_indices(self, text:str, substring:str):
        start = text.find(substring)
        if start == -1:
            return None
        end = start+len(substring)
        return start, end