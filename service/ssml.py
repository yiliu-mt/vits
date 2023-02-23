import logging
import traceback
from xml.dom import minidom
from .cn_tn import TextNorm


def parse_ssml(ssml, text_normalizer=None):
    try:
        if text_normalizer is None:
            text_normalizer = TextNorm(remove_punc=False)
        dom = minidom.parseString(ssml)
        dom: minidom.Document
        speak_elements = dom.getElementsByTagName('speak')
        speak_element = speak_elements[0]
        speak_element: minidom.Element
        parse_text = []
        for node in speak_element.childNodes:
            if isinstance(node, minidom.Text):
                parse_text.append(text_normalizer(node.nodeValue))
            elif node.tagName == "break":
                if "time" in node.attributes:
                    parse_text.append(node.attributes['time'].value)
                else:
                    parse_text.append('sp1')
            elif node.tagName == "sub":
                if "alias" in node.attributes:
                    parse_text.append(node.attributes['alias'].value)
                else:
                    logging.warn("no [alias] value in [sub], this element is useless")
            elif node.tagName == "phoneme":
                if "ph" in node.attributes:
                    parse_text.append(node.attributes['ph'].value)
                else:
                    logging.warn("no [ph] value in [phoneme], this element is useless")
            else:
                logging.warning("Warning unknown node {}".format(node))
        parse_text = " ".join(parse_text)
        return [parse_text]
    except Exception as e:
        traceback.print_exc()
        logging.error("Failed to parse ssml [{}]: {}".format(ssml, str(e)))
        return []
