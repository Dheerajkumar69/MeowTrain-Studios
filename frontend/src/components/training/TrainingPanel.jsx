/**
 * TrainingPanel — thin orchestrator that delegates to:
 *   • ConfigForm   – training configuration (idle state)
 *   • LiveMonitor  – progress / gauges / chart (training state)
 *   • ExportPanel  – GGUF export after training completes
 *
 * All gauge primitives live in ./Gauges.jsx.
 */
import { useState, useEffect, useRef } from 'react';
import { trainingAPI, hardwareAPI } from '../../services/api';
import { createWebSocket } from '../../services/ws';
import ConfirmDialog from '../ConfirmDialog';
import ConfigForm from './ConfigForm';
import LiveMonitor from './LiveMonitor';
import ExportPanel from './ExportPanel';

export default function TrainingPanel({ projectId, selectedModel, datasets, project, onProjectUpdate }) {
    const [config, setConfig] = useState({
        method: 'lora',
        epochs: 3,
        batch_size: 4,
        learning_rate: 2e-4,
        max_tokens: 512,
        train_split: 0.9,
        lora_rank: 16,
        lora_alpha: 32,
        lora_dropout: 0.05,
        warmup_steps: 10,
        gradient_accumulation_steps: 4,
        weight_decay: 0.01,
        lr_scheduler_type: 'cosine',
        early_stopping_patience: 3,
        early_stopping_threshold: 0.01,
        gradient_checkpointing: true,
        eval_steps: 50,
        dpo_beta: 0.1,
        orpo_alpha: 1.0,
        multi_gpu: false,
        deepspeed_stage: 2,
        resume_from_checkpoint: false,
    });

    const [training, setTraining] = useState(false);
    const [status, setStatus] = useState(null);
    const [hardware, setHardware] = useState(null);
    const [deviceInfo, setDeviceInfo] = useState(null);
    const [lossHistory, setLossHistory] = useState([]);
    const [error, setError] = useState('');
    const [wsError, setWsError] = useState(false);
    const [actionPending, setActionPending] = useState(null);
    const [showStopConfirm, setShowStopConfirm] = useState(false);

    const wsRef = useRef(null);
    const hwPollRef = useRef(null);
    const defaultsAppliedRef = useRef(false);

    const readyDatasets = datasets?.filter((d) => d.status === 'ready') || [];

    // ── Fetch device-recommended defaults on mount ──────────────
    useEffect(() => {
        let cancelled = false;
        const fetchDefaults = async () => {
            try {
                const res = await hardwareAPI.deviceInfo();
                if (cancelled) return;
                setDeviceInfo(res.data);
                // Merge recommended defaults into config only once
                if (!defaultsAppliedRef.current && res.data?.recommended_defaults) {
                    const rec = res.data.recommended_defaults;
                    defaultsAppliedRef.current = true;
                    setConfig(prev => ({
                        ...prev,
                        ...(rec.batch_size != null && { batch_size: rec.batch_size }),
                        ...(rec.max_tokens != null && { max_tokens: rec.max_tokens }),
                        ...(rec.gradient_accumulation_steps != null && { gradient_accumulation_steps: rec.gradient_accumulation_steps }),
                        ...(rec.gradient_checkpointing != null && { gradient_checkpointing: rec.gradient_checkpointing }),
                        ...(rec.method && { method: rec.method }),
                        // fp16/bf16 are internal — not exposed in config form
                    }));
                }
            } catch (err) { console.debug('Device info fetch failed (using defaults):', err.message || err); }
        };
        fetchDefaults();
        return () => { cancelled = true; };
    }, []);

    // ── Restore active training state on mount ──────────────────
    useEffect(() => {
        let cancelled = false;
        const checkActive = async () => {
            try {
                const res = await trainingAPI.status(projectId);
                if (cancelled) return;
                const s = res.data?.status;
                if (s === 'running' || s === 'paused' || s === 'training') {
                    setStatus(res.data);
                    setTraining(true);
                }
            } catch (err) { console.debug('No active training:', err.message || err); }
        };
        checkActive();
        return () => { cancelled = true; };
    }, [projectId]);

    // ── Hardware polling (always-on) ─────────────────────────────
    useEffect(() => {
        const poll = async () => {
            try {
                const res = await hardwareAPI.status();
                setHardware(res.data);
            } catch (err) { console.debug('HW poll:', err.message || err); }
        };
        poll();
        hwPollRef.current = setInterval(poll, 3000);
        return () => clearInterval(hwPollRef.current);
    }, []);

    // ── WebSocket for live training updates ──────────────────────
    useEffect(() => {
        if (!training) return;
        clearInterval(hwPollRef.current);

        const wsHandle = createWebSocket(
            projectId,
            (data) => {
                setWsError(false);
                setStatus(data);
                if (data.hardware) setHardware(data.hardware);
                if (data.current_loss != null) {
                    setLossHistory((prev) => [...prev.slice(-199), { step: data.current_step, loss: data.current_loss }]);
                }
                if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
                    setTraining(false);
                    onProjectUpdate?.();
                }
            },
            () => setWsError(true),
            () => { /* WS closed */ },
        );
        wsRef.current = wsHandle;

        return () => {
            wsHandle.close();
            const poll = async () => {
                try {
                    const res = await hardwareAPI.status();
                    setHardware(res.data);
                } catch (err) { console.debug('HW poll (cleanup):', err.message || err); }
            };
            hwPollRef.current = setInterval(poll, 3000);
        };
    }, [training, projectId]);

    // ── Actions ──────────────────────────────────────────────────
    const startTraining = async () => {
        if (!selectedModel) { setError('Please select a model first (Model tab).'); return; }
        if (readyDatasets.length === 0) { setError('Please upload training data first (Data tab).'); return; }
        setError('');
        setActionPending('starting');
        try {
            await trainingAPI.configure(projectId, { ...config, base_model: selectedModel.model_id });
            await trainingAPI.start(projectId);
            setTraining(true);
            setLossHistory([]);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start training');
        } finally {
            setActionPending(null);
        }
    };

    const pauseTraining = async () => {
        setActionPending('pausing');
        try { await trainingAPI.pause(projectId); }
        catch (err) { setError(err.response?.data?.detail || 'Pause failed'); }
        finally { setActionPending(null); }
    };

    const resumeTraining = async () => {
        setActionPending('resuming');
        try { await trainingAPI.resume(projectId); }
        catch (err) { setError(err.response?.data?.detail || 'Resume failed'); }
        finally { setActionPending(null); }
    };

    const stopTraining = async () => {
        setShowStopConfirm(false);
        setActionPending('stopping');
        try { await trainingAPI.stop(projectId); setTraining(false); }
        catch (err) { setError(err.response?.data?.detail || 'Stop failed'); }
        finally { setActionPending(null); }
    };

    // ── Render ────────────────────────────────────────────────────
    const isActive = training || status?.status === 'running' || status?.status === 'paused';

    if (isActive) {
        return (
            <>
                <LiveMonitor
                    status={status}
                    hardware={hardware}
                    lossHistory={lossHistory}
                    selectedModel={selectedModel}
                    wsError={wsError}
                    actionPending={actionPending}
                    onPause={pauseTraining}
                    onResume={resumeTraining}
                    onStop={() => setShowStopConfirm(true)}
                />
                <ConfirmDialog
                    open={showStopConfirm}
                    title="Stop Training?"
                    message="Progress will be saved as a checkpoint. You can resume from a checkpoint later."
                    confirmLabel="Stop Training"
                    variant="danger"
                    onConfirm={stopTraining}
                    onCancel={() => setShowStopConfirm(false)}
                />
            </>
        );
    }

    return (
        <>
            <ConfigForm
                config={config}
                setConfig={setConfig}
                selectedModel={selectedModel}
                readyDatasets={readyDatasets}
                hardware={hardware}
                deviceInfo={deviceInfo}
                projectId={projectId}
                error={error}
                setError={setError}
                actionPending={actionPending}
                onStartTraining={startTraining}
            />

            {/* Stop Training Confirmation Dialog */}
            <ConfirmDialog
                open={showStopConfirm}
                title="Stop Training?"
                message="Progress will be saved as a checkpoint. You can resume from a checkpoint later."
                confirmLabel="Stop Training"
                variant="danger"
                onConfirm={stopTraining}
                onCancel={() => setShowStopConfirm(false)}
            />

            {/* GGUF Export for LM Studio */}
            {(status?.status === 'completed' || project?.status === 'trained') && (
                <ExportPanel projectId={projectId} setError={setError} />
            )}
        </>
    );
}
