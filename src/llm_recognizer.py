import logging
from typing import Optional, List, Tuple, Set, Dict
import re
import json
import streamlit as st

from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    EntityRecognizer,
    AnalysisExplanation
)
from presidio_analyzer.nlp_engine import NlpArtifacts
from src.connectors.openai_connector import predict_with_gpt_4

logger = logging.getLogger("presidio-cli")

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
                check_label_groups: Optional[Tuple[Set, Set]] = None,
                custom_entities: Optional[Dict[str, Dict[str, str]]] = None):
        self.model_name = model_name
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )
        
        # Custom entities support
        self.custom_entities = custom_entities or {}
        
        # Add custom entities to supported entities
        self.supported_entities = supported_entities if supported_entities else self.ENTITIES
        if self.custom_entities:
            self.supported_entities = list(set(self.supported_entities + list(self.custom_entities.keys())))

        super().__init__(
            supported_entities=self.supported_entities,
            name=self.model_name
        )

    def get_supported_entities(self) -> List[str]:
        """Return list of supported entities including custom entities."""
        return self.supported_entities
    
    def load_model(self):
        pass

    def analyze(
            self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None,
            allow_list: Optional[List[str]] = None, ad_hoc_recognizers: Optional[List] = None, **kwargs
    ) -> List[RecognizerResult]:
        """
        Analyze the text and return a list of AnalysisExplanation objects.
        """
        if not entities:
            entities = self.supported_entities
        
        # Print all entities being searched for in a single list
        logger.info(f"LlmRecognizer searching for entities: {entities}")
        logger.info(f"LlmRecognizer custom entities: {self.custom_entities}")
        
        # Create the prompt
        prompt = (
            "Given below is a text input that might contain various types of Personal Identifiable Information (PII).\n"
            "Your task is to extract PII entities of the following types **ONLY**:\n"
        )
        
        # Add standard entities with basic description
        standard_entities = [e for e in entities if e not in self.custom_entities]
        if standard_entities:
            prompt += f"Standard entities: {', '.join(standard_entities)}\n"
        
        # Add custom entities with descriptions
        if any(e in self.custom_entities for e in entities):
            prompt += "\nCustom entity types to detect:\n"
            for entity in entities:
                if entity in self.custom_entities:
                    description = self.custom_entities[entity].get('description', 'No description provided')
                    examples = self.custom_entities[entity].get('examples', '')
                    prompt += f"- {entity}: {description}\n"
                    if examples:
                        prompt += f"  Examples: {examples}\n"
        
        prompt += (
            "\nFor each detected entity, output an object with the following keys:\n"
            "- 'entity_type': a string representing the type of PII (e.g. PERSON, LOCATION, EMAIL)\n"
            "- 'entity_text': a string representing the verbatim PII text, exactly as it appears in the input\n"
            "- 'score': a float between 0 and 1, representing the confidence in the detection\n"
            "- 'explanation': a string explaining the reasoning behind the detection\n"
        )
        
        # Add allowlist information to the prompt if provided
        if allow_list and len(allow_list) > 0:
            prompt += (
                "\nIMPORTANT: The following terms should NOT be considered PII, even if they match entity patterns:\n"
                f"{', '.join(allow_list)}\n"
            )
        
        # Complete the prompt
        prompt += (
            "\nMake sure your output is **ONLY** a valid JSON array containing these objects, and nothing else. "
            "No text before or after the JSON array. Just a valid JSON array.\n"
            "\nNow, extract the PII entities from the following text:\n"
            f"Text: {text}"
        )
        
        if "gpt_4o" in self.model_name:
            response_str = predict_with_gpt_4(prompt)
        else:
            raise ValueError("Invalid Model Name")
        
        logger.info(f"PROMPT from {self.model_name}: {prompt}")
        logger.info(f"RESPONSE STR from {self.model_name}: {response_str}")

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
        
        if 'llm_debug' in st.session_state:
            st.session_state.llm_debug['prompt'] = prompt
            st.session_state.llm_debug['response'] = response_str
        
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