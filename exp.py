def train(logger, **kwargs):
    logger.log("I am training")


def test(logger, **kwargs):
    logger.log("I am testing")


def echo(name, logger, **kwargs):
    logger.log(f"Echo: {name}")
