# Generated from resources/icons
import io

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

from PIL import ImageTk, Image

_CROSSHAIRS = b"""/* XPM */
static char * crosshairs_xpm[] = {
"16 16 2 1",
".	c #000000",
" 	c #FFFFFF",
"....... ........",
".....     ......",
"...   . .   ....",
"..  ... ...  ...",
".. ..     .. ...",
".  .  . .  .  ..",
". .. .. .. .. ..",
"               .",
". .. .. .. .. ..",
".  .  . .  .  ..",
".. ..     .. ...",
"..  ... ...  ...",
"...   . .   ....",
".....     ......",
"....... ........",
"................"};"""


_SHIFT = b"""/* XPM */
static char * shift_xpm[] = {
"16 16 2 1",
".	c #000000",
" 	c #FFFFFF",
"................",
"................",
"................",
".........    ...",
"...........  ...",
".......... . ...",
"......... .. ...",
"........ .......",
"....... ........",
"...... .........",
"..... ..........",
"... ............",
"..   ...........",
"... ............",
"................",
"................"};"""

_ZOOM = b"""/* XPM */
static char * zoom_xpm[] = {
"16 16 2 1",
".	c #000000",
" 	c #FFFFFF",
"................",
"........     ...",
".......  ...  ..",
"......  .....  .",
"...... ....... .",
"...... ....... .",
"...... ....... .",
"......  .....  .",
"..... .  ...  ..",
".... . .     ...",
"... . . ........",
".. . . .........",
". . . ..........",
".  . ...........",
"..  ............",
"................"};"""


def init():
    globs = globals()
    for k, v in list(globs.items()):
        if k.startswith('_') and isinstance(v, bytes) and v.startswith(b'/* XPM */') and k[1:] not in globs:
            globs[k[1:]] = ImageTk.BitmapImage(Image.open(io.BytesIO(v)).convert(mode='1'))

def get(name, **kw):
    return ImageTk.BitmapImage(Image.open(io.BytesIO(globals()['_' + name])).convert(mode='1'), **kw)
