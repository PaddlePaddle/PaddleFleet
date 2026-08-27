import sys
from typing import TYPE_CHECKING

from ..utils.lazy_import import _LazyModule

import_structure = {
    "utils": [
        "GenerationMixin",
        "MinLengthLogitsProcessor",
        "convert_dtype",
        "get_unfinished_flag",
        "LogitsProcessor",
        "BeamHypotheses",
        "RepetitionPenaltyLogitsProcessor",
        "LogitsProcessorList",
        "TopKProcess",
        "map_structure",
        "BeamSearchScorer",
        "TopPProcess",
        "get_scale_by_dtype",
        "validate_stopping_criteria",
        "apply_repetition_penalty",
        "sample_with_top_k",
        "sample_with_top_p",
    ],
    "config": ["FleetGenerationConfig"],
    "csa_cache": ["CSADynamicCache"],
    "greedy_generator": ["DynamicKVCache", "GreedyGenerator"],
    "inference_utils": ["init_inference_fleet"],
    "model_outputs": ["ModelOutput"],
    "configuration_utils": ["GenerationConfig", "resolve_hf_generation_config_path"],
    "logits_process": [
        "MinLengthLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "TopPProcess",
        "LogitsWarper",
        "HammingDiversityLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "ForcedBOSTokenLogitsProcessor",
        "LogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKProcess",
        "_get_ngrams",
        "_get_generated_ngrams",
        "LogitsProcessorList",
        "NoBadWordsLogitsProcessor",
        "_calc_banned_ngram_tokens",
    ],
    "stopping_criteria": [
        "validate_stopping_criteria",
        "StoppingCriteria",
        "MaxLengthCriteria",
        "StoppingCriteriaList",
        "MaxTimeCriteria",
    ],
    "streamers": ["BaseStreamer", "TextIteratorStreamer", "TextStreamer"],
}

if TYPE_CHECKING:
    from .config import FleetGenerationConfig
    from .configuration_utils import GenerationConfig
    from .csa_cache import CSADynamicCache
    from .greedy_generator import DynamicKVCache, GreedyGenerator
    from .inference_utils import init_inference_fleet
    from .logits_process import (
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        LogitsProcessor,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TopKProcess,
        TopPProcess,
    )
    from .stopping_criteria import (
        MaxLengthCriteria,
        MaxTimeCriteria,
        StoppingCriteria,
        StoppingCriteriaList,
        validate_stopping_criteria,
    )
    from .streamers import BaseStreamer, TextIteratorStreamer, TextStreamer
    from .utils import (
        BeamSearchScorer,
        GenerationMixin,
        apply_repetition_penalty,
        get_unfinished_flag,
        sample_with_top_k,
        sample_with_top_p,
    )
else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
