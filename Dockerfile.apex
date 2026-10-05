FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements-apex.txt .
RUN pip install --no-cache-dir -r requirements-apex.txt \
    && useradd --create-home --uid 10001 apex \
    && mkdir -p /app/data && chown apex:apex /app/data
COPY apex_bot ./apex_bot
USER apex
CMD ["python", "-m", "apex_bot"]
