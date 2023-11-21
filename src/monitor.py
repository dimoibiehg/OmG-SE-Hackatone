from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

DATA_TRACER = None
# Set up the tracer provider
trace.set_tracer_provider(TracerProvider())

# Set up the exporter (replace with your chosen exporter)
OLTP_EXPORTER = OTLPSpanExporter(endpoint="http://localhost:4320", insecure=True)

# Set up the span processor with the exporter
SPAN_PROCESSOR = BatchSpanProcessor(OLTP_EXPORTER)
trace.get_tracer_provider().add_span_processor(SPAN_PROCESSOR)


# Get a tracer
DATA_TRACER = trace.get_tracer(__name__)




