import logging

logger = logging.getLogger(__name__)


def print_grad(model):
    """
    called after loss.backward()
    """
    for name, parms in model.named_parameters():
        logger.info('--->name: %s', name)
        logger.info('--->grad_requirs: %s', parms.requires_grad)
        logger.info('--->grad_value: %s', parms.grad)
        # logger.info('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)