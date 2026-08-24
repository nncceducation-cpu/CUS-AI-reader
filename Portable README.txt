CUS AI Reader offline edition
=============================

System requirement
------------------
Windows 10 or Windows 11, 64-bit.

Start the app
-------------
1. Extract the complete ZIP to a normal folder. Do not run it from inside the ZIP.
2. Double-click "Start CUS AI Reader.cmd".
3. Wait for the browser to open at a 127.0.0.1 address.
4. Accept the research-use notice.

No internet connection or separate Python installation is required. Keep the runtime,
cus_ai, models, and .streamlit folders beside the launcher.

Stop the app
------------
Close the black launcher window or press Ctrl+C in that window.

If the browser does not open
----------------------------
Keep the launcher window open, then double-click "CUS AI Reader Local Page.url". You
can also paste http://127.0.0.1:8501 into Chrome or Edge. Startup details are normally
saved in %LOCALAPPDATA%\CUS-AI-reader\startup.log, with the Windows temporary folder
used as a fallback.

Data handling
-------------
The server listens only on this computer. Uploaded files are processed in the active
local session. The app has no database and does not intentionally save uploaded media.
Temporary video files are deleted after decoding. Reports are saved only if the user
clicks a download button and chooses a location.

Use de-identified files only. Burned-in names, identifiers, dates, faces, voice, and
screen-capture metadata can remain in ultrasound media even after DICOM tags are removed.

Clinical boundary
-----------------
Research and education use only. This prototype is not a medical device. It must not be
used for treatment or counselling decisions. No validated diagnostic model weights are
included. A qualified clinician must review the complete examination and every output.
