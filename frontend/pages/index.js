import Head from 'next/head'
import { useEffect, useMemo, useState } from 'react'

export default function Home() {
    const [message, setMessage] = useState('')
    const [history, setHistory] = useState([])
    const [loading, setLoading] = useState(false)
    const [streamingText, setStreamingText] = useState('')
    const [threadId, setThreadId] = useState('default')
    const [threadDraft, setThreadDraft] = useState('default')
    const [connectionError, setConnectionError] = useState('')

    const activeBubbles = useMemo(() => {
        const bubbles = [...history]
        if (loading && streamingText) {
            bubbles.push({ from: 'ai', text: streamingText, streaming: true })
        }
        return bubbles
    }, [history, loading, streamingText])

    useEffect(() => {
        if (typeof window === 'undefined') return

        const params = new URLSearchParams(window.location.search)
        const savedThreadId = params.get('thread_id') || 'default'
        setThreadId(savedThreadId)
        setThreadDraft(savedThreadId)
        loadThreadHistory(savedThreadId)
    }, [])

    async function loadThreadHistory(nextThreadId) {
        const activeThreadId = (nextThreadId || threadId || 'default').trim() || 'default'
        try {
            setConnectionError('')
            const res = await fetch(`http://127.0.0.1:8000/chat/history/${encodeURIComponent(activeThreadId)}`)
            if (!res.ok) {
                throw new Error(`Failed to load history: ${res.status}`)
            }
            const data = await res.json()
            const mapped = (data.messages || []).map((item) => ({
                from: item.role === 'assistant' ? 'ai' : item.role === 'user' ? 'user' : 'system',
                text: item.content,
            }))
            setHistory(mapped)
            setStreamingText('')
        } catch (e) {
            setConnectionError(String(e))
            setHistory([{ from: 'system', text: 'Could not load thread history: ' + String(e) }])
        }
    }

    useEffect(() => {
        if (typeof window === 'undefined') return
        const params = new URLSearchParams(window.location.search)
        params.set('thread_id', threadId)
        window.history.replaceState({}, '', `${window.location.pathname}?${params.toString()}`)
    }, [threadId])

    async function loadThread() {
        const nextThreadId = threadDraft.trim() || 'default'
        setThreadId(nextThreadId)
        await loadThreadHistory(nextThreadId)
    }

    async function send() {
        if (!message.trim()) return
        setConnectionError('')
        setHistory((h) => [...h, { from: 'user', text: message }])
        setStreamingText('')
        setLoading(true)
        try {
            const res = await fetch('http://127.0.0.1:8000/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message, thread_id: threadId, user_id: 'web_user' })
            })
            if (!res.ok || !res.body) {
                throw new Error(`Request failed with status ${res.status}`)
            }

            const reader = res.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''
            let finalText = ''

            while (true) {
                const { value, done } = await reader.read()
                if (done) break
                buffer += decoder.decode(value, { stream: true })

                let boundaryIndex = buffer.indexOf('\n\n')
                while (boundaryIndex !== -1) {
                    const event = buffer.slice(0, boundaryIndex).trim()
                    buffer = buffer.slice(boundaryIndex + 2)
                    boundaryIndex = buffer.indexOf('\n\n')

                    if (!event.startsWith('data:')) continue
                    const payload = JSON.parse(event.replace(/^data:\s*/, ''))

                    if (payload.token) {
                        finalText += payload.token
                        setStreamingText(finalText)
                    }

                    if (payload.error) {
                        throw new Error(payload.error)
                    }
                }
            }

            setHistory((h) => [...h, { from: 'ai', text: finalText || 'No reply.' }])
            setStreamingText('')
            setMessage('')
            await loadThreadHistory(threadId)
        } catch (e) {
            setConnectionError(String(e))
            setHistory((h) => [...h, { from: 'system', text: 'Error: ' + String(e) }])
        } finally {
            setLoading(false)
        }
    }

    return (
        <>
            <Head>
                <title>RAG Chatbot</title>
                <meta name="viewport" content="width=device-width, initial-scale=1" />
            </Head>

            <style jsx global>{`
                body {
                    background:
                        radial-gradient(circle at top left, rgba(13, 110, 253, 0.12), transparent 24%),
                        radial-gradient(circle at top right, rgba(32, 201, 151, 0.10), transparent 24%),
                        linear-gradient(180deg, #f6f8fb 0%, #eef2f7 100%);
                    min-height: 100vh;
                }
            `}</style>

            <div className="container py-4 py-lg-5" style={{ maxWidth: 1180 }}>
                <div className="row g-4 align-items-stretch">
                    <div className="col-12 col-lg-4">
                        <div className="card border-0 shadow-sm h-100">
                            <div className="card-body p-4">
                                <div className="d-flex align-items-center justify-content-between mb-3">
                                    <div>
                                        <div className="text-uppercase text-primary fw-semibold small">RAG Chatbot</div>
                                        <h1 className="h4 mb-0">Thread controls</h1>
                                    </div>
                                    <span className="badge text-bg-primary-subtle text-primary-emphasis">Beta</span>
                                </div>

                                <p className="text-secondary mb-4">
                                    This UI loads the chat history from the backend by <code>thread_id</code>.
                                    Refreshing the page will keep the same thread.
                                </p>

                                <label htmlFor="thread-id" className="form-label fw-semibold">
                                    Thread ID
                                </label>
                                <div className="input-group mb-3">
                                    <input
                                        id="thread-id"
                                        className="form-control"
                                        value={threadDraft}
                                        onChange={(e) => setThreadDraft(e.target.value)}
                                        placeholder="default"
                                    />
                                    <button className="btn btn-outline-primary" onClick={loadThread}>
                                        Load
                                    </button>
                                </div>

                                <div className="d-grid gap-2">
                                    <button
                                        className="btn btn-outline-secondary"
                                        onClick={() => {
                                            setHistory([])
                                            setStreamingText('')
                                            setConnectionError('')
                                            loadThreadHistory(threadId)
                                        }}
                                    >
                                        Reload current thread
                                    </button>
                                    <button
                                        className="btn btn-light border"
                                        onClick={() => {
                                            setHistory([])
                                            setStreamingText('')
                                            setMessage('')
                                        }}
                                    >
                                        Clear view
                                    </button>
                                </div>

                                <hr className="my-4" />

                                <div className="small text-secondary mb-2">Connection</div>
                                <div className={"badge " + (connectionError ? 'text-bg-danger' : 'text-bg-success')}>
                                    {connectionError ? 'Issue detected' : 'Connected'}
                                </div>
                                {connectionError && (
                                    <div className="alert alert-danger mt-3 py-2 small mb-0">
                                        {connectionError}
                                    </div>
                                )}

                                <div className="mt-4 p-3 rounded-3 bg-light border small text-secondary">
                                    <div className="fw-semibold text-dark mb-1">Notes</div>
                                    <ul className="mb-0 ps-3">
                                        <li>Messages are stored in SQLite on the backend.</li>
                                        <li>Streaming appears while the answer is generated.</li>
                                        <li>Same thread id = same previous chat.</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="col-12 col-lg-8">
                        <div className="card border-0 shadow-sm h-100">
                            <div className="card-header bg-white border-0 p-4 pb-2">
                                <div className="d-flex justify-content-between align-items-center flex-wrap gap-2">
                                    <div>
                                        <div className="text-muted small">Current thread</div>
                                        <h2 className="h5 mb-0">{threadId}</h2>
                                    </div>
                                    <span className="badge text-bg-light border text-secondary">
                                        Backend: /chat/stream
                                    </span>
                                </div>
                            </div>

                            <div className="card-body p-4">
                                <div
                                    className="rounded-4 border bg-body-tertiary p-3 p-md-4 mb-4"
                                    style={{ minHeight: 520, maxHeight: 620, overflowY: 'auto' }}
                                >
                                    {activeBubbles.length === 0 && (
                                        <div className="d-flex h-100 align-items-center justify-content-center text-secondary py-5">
                                            <div className="text-center">
                                                <div className="display-6 mb-2">💬</div>
                                                <div className="fw-semibold">Start a conversation</div>
                                                <div className="small">Ask something about ML / DL / maths.</div>
                                            </div>
                                        </div>
                                    )}

                                    <div className="d-flex flex-column gap-3">
                                        {activeBubbles.map((m, i) => {
                                            const isUser = m.from === 'user'
                                            const isSystem = m.from === 'system'
                                            return (
                                                <div key={i} className={"d-flex " + (isUser ? 'justify-content-end' : 'justify-content-start')}>
                                                    <div
                                                        className={
                                                            'px-3 py-2 rounded-4 shadow-sm ' +
                                                            (isUser
                                                                ? 'bg-primary text-white ms-5'
                                                                : isSystem
                                                                    ? 'bg-danger-subtle text-danger-emphasis me-5'
                                                                    : 'bg-white border me-5')
                                                        }
                                                        style={{ whiteSpace: 'pre-wrap', maxWidth: '82%', lineHeight: 1.5 }}
                                                    >
                                                        <div className="small opacity-75 mb-1 fw-semibold">
                                                            {isUser ? 'You' : isSystem ? 'System' : 'Assistant'}
                                                            {m.streaming ? ' · typing…' : ''}
                                                        </div>
                                                        <div>{m.text}</div>
                                                    </div>
                                                </div>
                                            )
                                        })}
                                    </div>
                                </div>

                                <div className="border-top pt-3">
                                    <label className="form-label fw-semibold">Send a message</label>
                                    <div className="input-group input-group-lg">
                                        <input
                                            className="form-control"
                                            value={message}
                                            onChange={(e) => setMessage(e.target.value)}
                                            placeholder="Type your question"
                                            onKeyDown={(e) => {
                                                if (e.key === 'Enter') send()
                                            }}
                                        />
                                        <button className="btn btn-primary px-4" onClick={send} disabled={loading}>
                                            {loading ? 'Sending…' : 'Send'}
                                        </button>
                                    </div>
                                    <div className="text-secondary small mt-2">
                                        Tip: press Enter to send.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
