FROM python:3.11

RUN apt-get update && apt-get install -y \
    pulseaudio \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . ./

# Add entry point to run the script
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
