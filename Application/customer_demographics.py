"""The existing statistics age bands and calendar-age convention."""

AGE_LIMITS = (17, 25, 35, 45, 55, 65)
AGE_LABELS = ("До 18 лет", "18–25 лет", "26–35 лет", "36–45 лет", "46–55 лет", "56–65 лет",
              "66 лет и старше", "Возраст не определён")


def calendar_age(born, as_of):
    if born is None or born > as_of:
        return None
    # As in statistics: a Feb 29 birthday is reached on Mar 1 in non-leap years.
    return as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))


def age_group(age):
    if age is None:
        return None
    return next((label for limit, label in zip(AGE_LIMITS, AGE_LABELS) if age <= limit), AGE_LABELS[-2])
