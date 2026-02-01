FROM busybox

# Copy models to a neutral internal location
COPY models/currentAiSolution.h5 /models/currentAiSolution.h5
COPY models/currentOlsSolution.pkl /models/currentOlsSolution.pkl
COPY models/scaler.pkl /models/scaler.pkl

COPY Readme.md /Readme.md

# On container start: copy models into the shared /tmp volume
CMD ["sh", "-c", "mkdir -p /tmp/knowledge/models && cp /models/* /tmp/knowledge/models/ && sleep 1"]
