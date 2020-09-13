Getting started
===============

Enviornment setup
-----------------

It's recommended that this software is installed in a virtualenv to avoid installing system-level
packages in the system's python installation. Many linux distros use python in their core components,
so messing around with system python can break stuff.

To install cvastrophoto in a virtualenv, first create and activate the virtualenv:

.. code:: bash

    mkdir -p ~/venv/astro
    virtualenv ~/venv/astro
    . ~/venv/astro/bin/activate

From then on, before using the script you just need to activate the virtualenv (the last line).

To build from source, first install all the prerequisites in the virtualenv.

To do so, some system-level packages may need to be installed, since some dependencies (like astropy, indi, opencv, etc)
may need the development headers to build. You will need sudo for these:

* libraw-dev
* cfitsio-dev
* liberfa-dev
* libwcs-dev
* opencv-dev

The exact name of those packages and command to install them will depend on your distro.

Then proceed to install python requirements on the virtualenv:

.. code:: bash

    pip install -r requirements.txt

That will download and install the main requirements.

Some python packages are optional and not in the requirements, but they'll enable some useful optimizations/features:

* opencv-python: for some optimizations and feature tracking during registration, optional but useful
* pyindi-client: for the guiding and capture stuff
* Imaging-tk: for the guiding GUI

Some distros don't include tkinter with their default python installation. If you want to use any GUI,
you will need it on the system. The virtualenv can't use it without it being present in the system python.
Install it through your package manager and not pip.

Building and installing cvastrophoto
------------------------------------

Activate the virtualenv, and build/install as usual:

.. code:: bash

    python setup.py build
    python setup.py develop

Using ``develop`` installs it into the virtualenv as a symlink to the source tree, so when you git pull you get
all the updates without having to reinstall. You can do that or you can install normally with ``setup.py install``,
your choice.

Test your installation with:

.. code:: bash

    python -m cvastrophoto --help
    python -m cvastrophoto apply somesourceimage.tiff somedestination.tiff abr:localgradient

If that works (regardless of whether it does something useful) odds are everything will work.

Basic stacking workflow
-----------------------

The way ``cvastrophoto`` is designed is to manage a global dark library that can be reused for many
sessions. So sessions don't have "darks" in them, as they are added to the dark library or taken from previously
acquired darks. Capture sessions should only have lights and flats.

So, first thing then is building a dark library.

Building a dark library
~~~~~~~~~~~~~~~~~~~~~~~

Gather all your darks, dark flats and bias frames into a subdirectory tree somewhere that works for your own
organization. It doesn't matter for ``cvastrophoto``, it will use the frame's EXIF or FITS headers to classify
them so the directory structure is meaningless.

.. important::

    Standard FITS headers for gain exist, but not for offset. If you don't always use the same offset for
    the same gain, odds are ``cvastrophoto`` will have a hard time figuring out how to match them, and may
    mix them in a single master dark. Best to avoid that, and always use the same offset for the same gain.

Then with a few commands you can build or update your library, always with the virtualenv activated:

.. code:: bash

    python -m cvastrophoto -j 6 darklib build -s Sources
    python -m cvastrophoto -j 6 biaslib build -s Sources

Now, ``-j 6`` tells it to use 6 threads. Stacking darks uses a median stacking method which is memory-hungry,
so depending on your system memory and number of subs, 6 threads may be too little or too much.
So if you've got RAM shortage, use a lower thread count. If your system is more powerful, conversely,
use a higher thread count. By default, it will use as many threads as processors are on your system, which may
be too many threads if you don't have the RAM for that level of parallelism.

That will create tons of master darks to be used as the need arises. It will classify images by temperature
and gain/ISO, camera name, brand and serial number, so it will be fully automated and safe to use.
It supports Canon raw and Nikon NEF files through libraw, FITS files common when capturing with INDI.

While ``cvastrophoto`` supports tiff/png files, using them in a dark library will be limiting because those
don't have astro-specific metadata embedded in them. It's better to stick to FITS or DSLR raw files.

The library will be placed on the default user-wide path in ``~/.cvastrophoto/darklib``. An alternate path
can be specified, but that default path is where all commands will go by default when trying to calibrate
frames, so it's better to build the library there.

After the library is built, the individual subs aren't necessary anymore, unless you intend to add frames
to them in the future and rebuild the library from scratch. Otherwise you can safely delete the sources after
buildling the library, and carry the library around instead, which should be much smaller.

Updating the library
--------------------

The same command used for building the library can be used to update it. If the reason to update the library
is to add new darks for, say, new gain or temperature settings, then the same command will do the trick:

.. code:: bash

    python -m cvastrophoto -j 6 darklib build -s Sources
    python -m cvastrophoto -j 6 biaslib build -s Sources

By default, the command only adds master darks that weren't present in the library, so it works for this purpose.

If, instead, the aim is to refresh the existing darks to account for noise pattern changes, something that should
be done periodically (every few months), then a refresh should be requested:

.. code:: bash

    python -m cvastrophoto -j 6 darklib build --refresh -s Sources
    python -m cvastrophoto -j 6 biaslib build --refresh -s Sources

This will recreate all master darks possible with the given sources, stepping on any existing master darks in the
library, but leaving existing ones without matching sources alone. Ie: it will refresh all it can without removing
anything.

Stacking a capture session
~~~~~~~~~~~~~~~~~~~~~~~~~~

The most convenient way of organizing session data so it can be effortlessly stacked by ``cvastrophoto`` is
to store all the lights and flats in a folder for that session specifically. For example:

.. code:: console

    claudiofreire@localhost:~/Pictures/Astro/SMC/NGC 346/QHY163m/2020-09-08> tree
    .
    ├── Flats
    │   ├── flat_001.fits
    │   └── flat_039.fits
    ├── Lights
    │   ├── light_028.fits
    │   └── light_052.fits

The default paths for ``Lights``, ``Flats``, ``Darks`` and ``Dark Flats`` don't need to be specified on the
command line so it makes it more convenient. As mentioned, unless your session had some specific needs,
you shouldn't have ``Darks`` or ``Dark Flats`` as using the library is preferred.

If you had a light leak or some other condition that requires custom calibration for this session, including
those directories will make ``cvastrophoto`` build a local dark library with them instead of using the global
library. This may be useful in some situations, but should not be the norm.

The ``process`` command takes care of stacking. Before proceeding, check that all frames have a matching dark
in the library by using ``--dark-annot`` on the data set:

.. code:: bash

    python -m cvastrophoto process -b 1 output.tiff --dark-annot

This will print the classification of each flat/light and whether it found a dark and bias file for it.
If it didn't for some, it will spit a summary of which are missing.

If may look a bit cryptic, but the gist of the matter is, if it doesn't report "biasless" or "darkless" subs,
you're fine. If it does, then you need to take new darks/bias.

If all is well, you're ready to stack. The defaults work out of the box for OSC/DSLR data:

.. code:: bash

    python -m cvastrophoto process -b 1 output.tiff

If you have mono data, the default bayer drizzle algorithm will probably fail. You need to specify an alternative
algorithm suitable for mono data, and the closest match would be ``adaptive``, which does the same kind of
outlier rejection without the bayer drizzling part:

.. code:: bash

    python -m cvastrophoto process -b 1 -m adaptive output.tiff

The command will take time. It will have to perform 3 passes through the subs, the adaptive stacking method is
an iterative method, and it will do registration on the first pass, which is quite CPU and memory intense.
Sit back and let it work. If you're impatient like me, you can request for a preview at intervals with a simple
change to the command:

.. code:: bash

    python -m cvastrophoto process -b 1 -m adaptive output.tiff -Pb 64 -P

Here, ``-Pb 64`` is the preview brightness (stretching). I like stretching previews aggressively to look into the
faint details and how they're coming along, ignoring the bright parts. Other brightness settings may suit
other preview needs. In any case only basic linear stretching is recommended for previews, because other stretching
methods are slow and not worth the wait. That argument alone won't turn on previews, the ``-P`` argument just requests
that it outputs a preview into ``preview-X.jpg`` with X being the integration phase it's at.

Adaptive has 3 phases: initial estimate, outlier rejection and final average (0 1 2 respectively).
Just let it work and open/refresh the preview periodically to see how it's going. Requesting for previews does
make the process slower since it will apply ABR and WB periodically to do so, and that takes time.

Without the preview, it goes substantially faster. It's a tradeoff.

Batch processing
~~~~~~~~~~~~~~~~

While the above may sound daunting, it's really easy to use once you get the hang of it.
And having all these commands in a script make it easy to revisit old projects and tweak things when you
refine your workflow.

For instance, this is a script I made to process a whole mono LRGB session in one go:

.. code:: bash

    set -e

    export MAX_MEMORY_OVERHEAD=4

    python -m cvastrophoto process -Pb 16 -b 1 --flatsdir Flats.L3 --lightsdir Lights.L --cache .cvapstatecache/Lf3 -Ri norm:fullstat -mw snr -m adaptive -ms localgradient:scale=512 ngc346_l.tiff &
    python -m cvastrophoto process -Pb 16 -b 1 --flatsdir Flats.R3 --lightsdir Lights.R --cache .cvapstatecache/Rf3 -Ri norm:fullstat -mw snr -m adaptive -ms localgradient:scale=512 ngc346_r.tiff &
    python -m cvastrophoto process -Pb 16 -b 1 --flatsdir Flats.G3 --lightsdir Lights.G --cache .cvapstatecache/Gf3 -Ri norm:fullstat -mw snr -m adaptive -ms localgradient:scale=512 ngc346_g.tiff &
    python -m cvastrophoto process -Pb 16 -b 1 --flatsdir Flats.B3 --lightsdir Lights.B --cache .cvapstatecache/Bf3 -Ri norm:fullstat -mw snr -m adaptive -ms localgradient:scale=512 ngc346_b.tiff &

    wait

    python -m cvastrophoto combine --mode slum ngc346_slum.tiff ngc346_{l,r,g,b}.tiff

    python -m cvastrophoto apply ngc346_slum.tiff ngc346_slum_nr.tiff nr:starlessdiffusion:L=0.25 &
    python -m cvastrophoto apply ngc346_r.tiff ngc346_r_nr.tiff nr:starlessdiffusion:L=0.45 &
    python -m cvastrophoto apply ngc346_g.tiff ngc346_g_nr.tiff nr:starlessdiffusion:L=0.45 &
    python -m cvastrophoto apply ngc346_b.tiff ngc346_b_nr.tiff nr:starlessdiffusion:L=0.45 &

    wait

    python -m cvastrophoto combine --mode lrgb --color-rops color:wb:wb_set=qhy163m-rgb -- ngc346_lrgb_nr.tiff ngc346_{slum,r,g,b}_nr.tiff
    python -m cvastrophoto apply ngc346_lrgb_nr.tiff ngc346_lrgb_nr_hdr.tiff neutralization:bg:scale=64 abr:localgradient:scale=512 stretch:hdr:steps=4

Here, we use job control to execute independent tasks in parallel. Each filter is stored in its own folder, not using
the defaults but a variation of sorts. We provide an explicit cache path to prevent conflicts, otherwise subs from
one filter could use cached results from another.

I also tweaked the ABR algorithm to specify a larger feature scale, something that preserves more detail if
the gradients aren't particularly complex.

The subs are normalized with the ``fullstat`` normalization operator, that normalizes sky levels and improves
outlier rejection, and weighed with the ``snr`` metric which measures SNR, accounting for varying levels of light
pollution in this case. The SNR metric also handles properly varying exposure settings, weighing each sub based on
its effective SNR and normalizing sky levels allows mixing different gain/ISO settings in a single stack.

Then the script proceeds to combining the lumminance and RGB filters into a superluminance channel, apply NR on each
independently with adequate parameters, and the combine the noise-reduced output into an LRGB color image.

The final step does background neutralization and another iteration of ABR (since background neutralization tends
to leave a bright background), and finally an HDR stretch.

With this script at hand, it's easy to make changes to the processing flow, tweak parameters and find the processing
pipeline that works best for each image. Rerunning the script won't re-register and stack all frames, since the stack
cache will contain pre-stacked raw data, keeping the iteration process quick.
