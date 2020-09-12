from __future__ import absolute_import

from collections import defaultdict
from queue import Queue
import numpy
import tempfile

from cvastrophoto.image import rgb


class MockDevice:

    def __init__(self, name):
        self.name = name
        self.properties = {}


class MockCCD(MockDevice):

    def __init__(self, name, w, h, cfa):
        super(MockCCD, self).__init__(name)
        self.blobs = defaultdict(Queue)
        self.t_shot_count = 0
        self.t_exposures = []
        self.t_frame_types = []
        self.t_upload_settings = []
        self.frame_type = None
        self.setUploadSettings('.', 'IMAGE_XXX')

        self._shape = (h, w)
        self.setUploadClient()

    def expose(self, exposure):
        self.t_exposures.append(exposure)
        self.t_frame_types.append(self.frame_type)
        self.t_upload_settings.append(self.properties["UPLOAD_SETTINGS"])
        self.t_shot_count += 1

        newblob = numpy.empty(self._shape, dtype=numpy.uint16)
        newblob[:] = min(65535, exposure * 100)

        if self.upload_mode == 'client':
            self.pushBLOB('CCD1', newblob)
        else:
            self.saveBLOB(newblob)

    def saveBLOB(self, blob):
        pass

    def pushBLOB(self, name, blob):
        self.blobs[name].put_nowait(blob)

    def pullBLOB(self, name):
        return self.blobs[name].get_nowait()

    def pullImage(self, name):
        return self.blob2Image(self.pullBLOB(name))

    def blob2Image(self, blob):
        return rgb.RGB(None, img=blob, linear=True, autoscale=False)

    def setUploadClient(self):
        self.upload_mode = 'client'

    def setUploadLocal(self):
        self.upload_mode = 'local'

    def setUploadMode(self, mode):
        self.upload_mode = mode

    def setLight(self):
        self.frame_type = 'Light'

    def setFlat(self):
        self.frame_type = 'Flat'

    def setDark(self):
        self.frame_type = 'Dark'

    def setUploadSettings(self, upload_dir=None, image_prefix=None, image_type=None, image_suffix=None):
        if not upload_dir:
            upload_dir = self.properties["UPLOAD_SETTINGS"][0]
        if not image_prefix and not image_type:
            image_pattern = self.properties["UPLOAD_SETTINGS"][1]
        else:
            image_pattern = '_'.join(filter(bool, [image_prefix, image_type, image_suffix, 'XXX']))
        self.setText("UPLOAD_SETTINGS", [upload_dir, image_pattern])

    def setProperty(self, name, values):
        self.properties[name] = values

    setText = setProperty
    setNumber = setProperty
    setSwitch = setProperty
