# usage:
# change example_input.txt to your input file on line 14
# docker build -t subwiz .
# docker run subwiz (optionally add any subwiz arguments)

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY subwiz subwiz
COPY example_input.txt input_domains.txt

ENTRYPOINT ["python", "-m", "subwiz.cli", "-i", "/app/input_domains.txt"]
CMD []
