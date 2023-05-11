import os
import logging
from utils import read_wav


def init_tts_resources(service_config_dict, vits_config):
    if 'recording_resource' in service_config_dict:
        logging.warning("Loading recording resources")
        service_config_dict['recording_resource_map'] = {}
        for resource_name, resource_path in service_config_dict["recording_resource"].items():
            if not os.path.exists(resource_path):
                continue
            service_config_dict['recording_resource_map'][resource_name], _ = read_wav(
                resource_path, vits_config.data.sampling_rate, vits_config.data.max_wav_value
            )

            