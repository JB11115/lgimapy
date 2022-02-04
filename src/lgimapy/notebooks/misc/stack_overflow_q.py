from statistics import mean, median


class Player:
    def __init__(self, name, age):
        self.name = name
        self._age = age

    @property
    def age(self):
        return self._age


class Team:
    def __init__(self, players):
        self.players = players

    @property
    def all_ages(self):
        return {player.name: player.age for player in self.players}

    @property
    def total_age(self):
        return sum([player.age for player in self.players])

    @property
    def average_age(self):
        return mean([player.age for player in self.players])

    @property
    def median_age(self):
        return median([player.age for player in self.players])


p1 = Player("John", 15)
p2 = Player("Tim", 17)
p3 = Player("Annie", 22)

team = Team([p1, p2, p3])
team.all_ages
team.total_age
team.median_age
team.average_age
