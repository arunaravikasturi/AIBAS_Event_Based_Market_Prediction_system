FROM busybox

# Copy models into image
COPY models/currentAiSolution.h5 /models/currentAiSolution.h5
COPY models/currentOlsSolution.pkl /models/currentOlsSolution.pkl
COPY models/ols_features.pkl /models/ols_features.pkl
COPY models/scaler.pkl /models/scaler.pkl

# On container start, copy everything into shared volume
CMD ["sh", "-c", "mkdir -p /tmp/knowledge/models && cp /models/* /tmp/knowledge/models/ && sleep 1"]
