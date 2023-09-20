FROM python:3.11
ADD . .
RUN pip3 install -r requirments.txt
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./train.py"]