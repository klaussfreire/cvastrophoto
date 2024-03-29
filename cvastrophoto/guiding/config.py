from __future__ import absolute_import

import ast
import logging


logger = logging.getLogger(__name__)


class ConfigHelperMixin(object):

    PARAM_TYPES = (bool, int, float)

    def load_config(self, config_file, section):
        if not config_file.has_section(section):
            return

        for opt in config_file.options(section):
            if not opt.startswith('_') and hasattr(self, opt) and isinstance(getattr(self, opt), self.PARAM_TYPES):
                value = ast.literal_eval(config_file.get(section, opt))
                setattr(self, opt, value)
                logger.info("Set guider %r to %r", opt, value)
