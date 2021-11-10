class Calendar(object):
    def __init__(self, year, month, day):
        self.year = str(year)
        self.month = '0'*(2 - len(str(month))) + str(month)
        self.day = '0'*(2 - len(str(day))) + str(day)
    def __str__(self):
        return self.day + '/' + self.month + '/' + self.year
    def __repr__(self):
        return self.__str__()
    def __add__(self,days):
        dayHold = int(self.month)*30 + int(self.day) + int(days)
        return Calendar(int(self.year) + dayHold//360, dayHold%360//30 + 1, dayHold%360%30)
    def __sub__(self,days):
        return self + (-1)*days
    def __eq__(self, other):
        return self.year == other.year and self.month == other.month and self.day == other.day
    def __ne__(self, other):
        return not self == other
