from datetime import datetime
from common.config import logger
#TODO: universal list of events

class Event:
    def __init__(self, date: str, label: str, metadata: dict = None, color=None):
        """
        Initialize an Event object.

        :param date: The date of the event in "YYYY-MM-DD" format.
        :param label: The label for the event (e.g., "MOASS").
        :param metadata: Optional dictionary for additional event details.
        :param color: Hex or (matplotlib) string colour.
        """
        self.label = label
        self.date = datetime.strptime(date, "%Y-%m-%d")  # Convert string to datetime if needed
        self.__metadata = metadata if metadata is not None else {}
        self.__color = color
        logger.debug(self.__repr__()+' created.')

    def __str__(self):
        return f"Event(label='{self.label}', date='{self.date}', metadata={self.__metadata}, color={self.__color})"

    def get_date_str(self):
        """
        Get the event date as a string in "YYYY-MM-DD" format.
        """
        return self.date.strftime("%Y-%m-%d")

    def diff(self, other_date: str):
        #TODO: update this
        """
        Calculate the number of days between an arbitrary date and the event date.
        :param other_date: The other date in "YYYY-MM-DD" format.
        :return: Number of days (negative if event date is later, positive otherwise).
        """
        other_date_obj = datetime.strptime(other_date, "%Y-%m-%d")
        difference = other_date_obj - self.date
        return difference.days

    #TODO: compatibility with upcoming ContinuousSignal calendar operations

    @property
    def color(self):
        return self.__color

    @property
    def metadata(self):
        return self.__metadata