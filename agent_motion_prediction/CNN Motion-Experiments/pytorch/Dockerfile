FROM tensorflow/tensorflow
ADD . .
RUN pip3 install -r tf_requirments.txt
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./train.py"]