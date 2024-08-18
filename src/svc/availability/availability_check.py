import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec


class ReservationService(BaseToolSpec):
    """
    A service to manage and check the availability of reservations for businesses.

    Attributes:
        business_availability (dict): Stores the availability data for businesses.
        opening_hours (dict): Stores the parsed opening hours for each business.
        business_dict (dict): Stores the mapping of business names to business IDs.
    """

    spec_functions = [
        "check_availability",
        "check_availability_range",
        "get_business_id",
    ]

    def __init__(
        self,
        opening_hours: Dict[str, Dict[str, Optional[str]]],
        business_dict: Dict[str, str],
    ):
        """
        Initializes the ReservationService with the given opening hours and business dictionary.

        Args:
            opening_hours (dict): A dictionary where the key is the business_id and the value is a dict of opening hours by day.
            business_dict (dict): A dictionary where the key is the biz_name and the value is the business_id.
        """
        self.business_availability = {}
        self.opening_hours = self.parse_opening_hours(opening_hours)
        self.business_dict = business_dict

    def parse_opening_hours(
        self, opening_hours: Dict[str, Dict[str, Optional[str]]]
    ) -> Dict[str, Dict[str, Dict[str, Optional[int]]]]:
        """
        Parses the opening hours dictionary into a more usable format.

        Args:
            opening_hours (dict): A dictionary where the key is the business_id and the value is a dict of opening hours by day.

        Returns:
            dict: A parsed dictionary of opening hours.
        """
        parsed_hours = {}
        for business_id, hours in opening_hours.items():
            parsed_hours[business_id] = {}
            for day, time_range in hours.items():
                if time_range is None:
                    parsed_hours[business_id][day] = {"open": None, "close": None}
                else:
                    open_close = time_range.split("-")
                    parsed_hours[business_id][day] = {
                        "open": int(open_close[0].split(":")[0]),
                        "close": int(open_close[1].split(":")[0]),
                    }
        return parsed_hours

    def is_open(self, business_id: str, date_time: str) -> bool:
        """
        Checks if a business is open at the given date and time.

        Args:
            business_id (str): The `business_id` value of the busienss, typically
            date_time (str): The date and time to check in ISO format (e.g., "2023-08-15T08:00:00").

        Returns:
            bool: True if the business is open, False otherwise.
        """
        date_time_obj = datetime.fromisoformat(date_time)
        day_name = date_time_obj.strftime("%A")
        if (
            business_id in self.opening_hours
            and day_name in self.opening_hours[business_id]
        ):
            open_hour = self.opening_hours[business_id][day_name]["open"]
            close_hour = self.opening_hours[business_id][day_name]["close"]
            if open_hour is None or close_hour is None:
                return False
            current_hour = date_time_obj.hour
            return open_hour <= current_hour < close_hour
        return False

    def hash_to_slots(self, business_id: str, date_time: str) -> int:
        """
        Generates a deterministic number of available slots using a hash of the business ID and date-time.

        Args:
            business_id (str): The ID of the business.
            date_time (str): The date and time to check in ISO format (e.g., "2023-08-15T08:00:00").

        Returns:
            int: The number of available slots (0 50% of the time, 1-10 otherwise).
        """
        date_time_obj = datetime.fromisoformat(date_time)
        seed_str = f"{business_id}_{date_time_obj.strftime('%Y-%m-%d_%H')}"
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)
        random.seed(seed)
        # 50% chance to return 0
        if seed % 2 == 0:
            return 0
        return random.randint(1, 10)

    def check_availability(self, business_id: str, date_time: str, pax: int) -> bool:
        """
        Checks if there are enough available slots for a given business, date-time, and number of pax.

        Args:
            business_id (str): The ID of the business.
            date_time (str): The date and time to check in ISO format (e.g., "2023-08-15T08:00:00").
            pax (int): The number of people.

        Returns:
            bool: True if there are enough available slots, False otherwise.
        """
        if not self.is_open(business_id, date_time):
            return False
        available_slots = self.hash_to_slots(business_id, date_time)
        return available_slots >= pax

    def check_availability_range(
        self, business_id: str, start_datetime: str, end_datetime: str, pax: int
    ) -> str:
        """
        Checks availability for a range of date-times at the hour level, considering
        the business's opening hours and the required number of pax.

        Args:
            business_id (str): The ID of the business.
            start_datetime (str): The start of the date range in ISO format (e.g., "2023-08-15T08:00:00").
            end_datetime (str): The end of the date range in ISO format (e.g., "2023-08-15T12:00:00").
            pax (int): The number of people.

        Returns:
            str: A JSON string describing the availability for each hour in the range.
        """
        availability = {}
        start_datetime_obj = datetime.fromisoformat(start_datetime)
        end_datetime_obj = datetime.fromisoformat(end_datetime)
        current_datetime = start_datetime_obj

        while current_datetime <= end_datetime_obj:
            current_datetime_str = current_datetime.isoformat()
            if self.is_open(business_id, current_datetime_str):
                datetime_str = current_datetime.strftime("%Y-%m-%d %H:00")
                available_slots = self.hash_to_slots(business_id, current_datetime_str)
                # Check if the available slots are sufficient for the pax size
                availability[datetime_str] = available_slots >= pax
            else:
                availability[current_datetime.strftime("%Y-%m-%d %H:00")] = False
            current_datetime += timedelta(hours=1)

        return json.dumps(availability, indent=4)

    def get_business_id(self, biz_name: str) -> Optional[str]:
        """
        Retrieves the business ID for a given business name.

        Args:
            biz_name (str): The name of the business.

        Returns:
            str: The business ID if found, None otherwise.
        """
        return self.business_dict.get(biz_name)
