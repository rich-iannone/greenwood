class CensoringType(str, Enum):
    """The censoring flavor of a `Surv` response."""

    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"
    COUNTING = "counting"


