FROM python:3.11

EXPOSE 8085

WORKDIR /app

COPY . ./

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx 

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8085", "--server.address=0.0.0.0"]

