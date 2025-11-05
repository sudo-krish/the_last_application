import asyncio, socket, time
import nodriver as uc
from src.utils import get_attributes
from pathlib import Path
import yaml
ROOT_DIR = Path(__file__).resolve().parent.parent
CSS_SELECTOR_PATH = ROOT_DIR / "config" / "css_selectors.yaml"
with open(CSS_SELECTOR_PATH, "r") as f:
    css_selectors = yaml.safe_load(f)
async def get_job_info(page, card):
    card_element = await card.query_selector(css_selectors["job_card"]["job_card_wrapper"])  # Adjust selector as needed
    job_info = {}
    job_attributes = get_attributes(card_element)
    job_info["job_id"] = job_attributes.get(css_selectors["job_card"]["job_id"], None)
    job_link = await card_element.query_selector("a")
    job_info["job_link"] = get_attributes(job_link).get("href", None) if job_link else None
    title_elem = await card_element.query_selector(css_selectors["job_card"]["job_title"])
    company = await card_element.query_selector(css_selectors["job_card"]["job_company"])
    location = await card_element.query_selector(css_selectors["job_card"]["job_location"])

    job_info["title"] = title_elem.text if title_elem else None
    job_info["company"] = company.text if company else None
    job_info["location"] = location.text if location else None

    hirer_information = await page.query_selector(css_selectors["hirer"]["hirer_information"])
    if hirer_information:
        hirer_profile_link = await hirer_information.query_selector(css_selectors["hirer"]["hirer_profile_link"])
        job_info["hirer_name"] = hirer_information.text if hirer_information else None
        job_info["hirer_profile_link"] = get_attributes(hirer_profile_link).get("href", None)
    else:
        job_info["hirer_name"] = None
        job_info["hirer_profile_link"] = None

    job_details = await page.query_selector_all(css_selectors["job_details"]["job_description_container"])
    job_description = "\n".join([d.text for d in job_details]) if job_details else None
    job_info["description"] = job_description

    return job_info

def validate_job_info(job_info):
    required_fields = ["job_id", "title", "company", "location", "job_link"]
    for field in required_fields:
        if field not in job_info or not job_info[field]:
            return False
    return True