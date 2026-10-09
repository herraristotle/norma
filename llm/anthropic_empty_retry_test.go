package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// anthropicEmptyStream mirrors the degenerate completion some gateways return:
// a valid message_start, an empty text block, and end_turn with zero content —
// the exact empty response seen in production through a Claude-compatible router.
var anthropicEmptyStream = sse(
	`event: message_start`,
	`data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"x","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
	``,
	`event: content_block_start`,
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
	``,
	`event: content_block_stop`,
	`data: {"type":"content_block_stop","index":0}`,
	``,
	`event: message_delta`,
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":0}}`,
	``,
	`event: message_stop`,
	`data: {"type":"message_stop"}`,
	``,
)

// anthropicRecoveredStream is a well-formed non-empty completion.
var anthropicRecoveredStream = sse(
	`event: message_start`,
	`data: {"type":"message_start","message":{"id":"m2","type":"message","role":"assistant","model":"x","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}`,
	``,
	`event: content_block_start`,
	`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
	``,
	`event: content_block_delta`,
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"recovered"}}`,
	``,
	`event: content_block_stop`,
	`data: {"type":"content_block_stop","index":0}`,
	``,
	`event: message_delta`,
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
	``,
	`event: message_stop`,
	`data: {"type":"message_stop"}`,
	``,
)

// A streamed empty completion is re-requested; a subsequent non-empty attempt
// surfaces normally. Exactly two requests: one empty, one recovered.
func TestAnthropicStreamEmptyResponseRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Write([]byte(anthropicEmptyStream))
			return
		}
		w.Write([]byte(anthropicRecoveredStream))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", EmptyResponseInterval: time.Millisecond})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "recovered" {
		t.Fatalf("text=%q", msg.Text())
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d, want 2 (one empty + one retry)", calls.Load())
	}
}

// An always-empty stream is retried a bounded number of times, then surfaced as
// an empty message rather than looping forever.
func TestAnthropicStreamEmptyResponseExhausts(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(anthropicEmptyStream))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", EmptyResponseInterval: time.Millisecond})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if len(msg.Content) != 0 {
		t.Fatalf("want empty message after exhaustion, got %+v", msg.Content)
	}
	if calls.Load() != int32(emptyResponseRetries+1) {
		t.Fatalf("calls=%d, want %d", calls.Load(), emptyResponseRetries+1)
	}
}

// A max_tokens truncation with no content must NOT retry — that recovery is the
// harness's to drive, and re-requesting would fight it.
func TestAnthropicStreamEmptyMaxTokensNoRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(sse(
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"m3","type":"message","role":"assistant","model":"x","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":0}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		)))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", EmptyResponseInterval: time.Millisecond})
	_ = collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if calls.Load() != 1 {
		t.Fatalf("calls=%d, want 1 (max_tokens must not retry)", calls.Load())
	}
}

// A stream that produces content on the first attempt is never retried, and its
// events reach the consumer in protocol order (message_start before content).
func TestAnthropicStreamContentNoRetry(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Write([]byte(sse(
			`event: message_start`,
			`data: {"type":"message_start","message":{"id":"m4","type":"message","role":"assistant","model":"x","content":[],"usage":{"input_tokens":3,"output_tokens":1}}}`,
			``,
			`event: content_block_start`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			``,
			`event: content_block_delta`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
			``,
			`event: content_block_stop`,
			`data: {"type":"content_block_stop","index":0}`,
			``,
			`event: message_delta`,
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			``,
			`event: message_stop`,
			`data: {"type":"message_stop"}`,
			``,
		)))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", EmptyResponseInterval: time.Millisecond})
	msg := collect(t, p, CompletionRequest{Messages: []Message{UserText("go")}})
	if msg.Text() != "hi" || calls.Load() != 1 {
		t.Fatalf("text=%q calls=%d", msg.Text(), calls.Load())
	}
}

// The non-streaming path retries an empty completion the same way.
func TestAnthropicCompleteEmptyResponseRetries(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("content-type", "application/json")
		if calls.Add(1) == 1 {
			w.Write([]byte(`{"id":"m5","type":"message","role":"assistant","model":"x","content":[],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":0}}`))
			return
		}
		w.Write([]byte(`{"id":"m6","type":"message","role":"assistant","model":"x","content":[{"type":"text","text":"recovered"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`))
	}))
	defer srv.Close()
	p, _ := NewProvider(Config{Format: FormatAnthropic, BaseURL: srv.URL, APIKey: "k", Model: "m", EmptyResponseInterval: time.Millisecond})
	msg, _, _, err := p.Complete(context.Background(), CompletionRequest{Messages: []Message{UserText("go")}})
	if err != nil {
		t.Fatalf("Complete err: %v", err)
	}
	if msg.Text() != "recovered" {
		t.Fatalf("text=%q", msg.Text())
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d, want 2", calls.Load())
	}
}
