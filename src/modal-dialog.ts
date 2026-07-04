export class ModalDialog {
    private overlay: HTMLElement;
    private modalContent: HTMLElement;
    private static instance: ModalDialog;
    private resizeObserver?: ResizeObserver;

    private constructor() {
        this.overlay = document.getElementById('contact-modal-overlay') as HTMLElement;
        this.modalContent = this.overlay.querySelector('.modal-content') as HTMLElement;

        if (!this.overlay || !this.modalContent) {
            throw new Error('Modal elements not found in the DOM');
        }

        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hide();
            }
        });

        window.addEventListener('popstate', (event) => {
            // If the modal is visible and the history state doesn't indicate it should be, hide it.
            if (this.overlay.style.display !== 'none' && !event.state?.modalOpen) {
                this.hide(true); // Hide without affecting history
            }
        });

        this.modalContent.addEventListener('click', (e) => {
            const target = e.target as HTMLElement;
            if (target.id === 'ok-button') {
                this.hide();
            }
        });

        this.modalContent.addEventListener('submit', async (e) => {
            if (!(e.target instanceof HTMLFormElement) || !e.target.classList.contains('contact-form')) {
                return;
            }

            e.preventDefault();

            const form = e.target;
            const status = form.querySelector('.form-status') as HTMLElement;
            if (!status) return;

            const data = new FormData(form);
            status.innerHTML = 'Sending...';
            status.className = 'form-status sending';

            try {
                const response = await fetch(form.action, {
                    method: form.method,
                    body: data,
                    headers: {
                        'Accept': 'application/json'
                    }
                });

                if (response.ok) {
                    form.reset();
                    // Disconnect the observer since the textarea is about to be removed.
                    if (this.resizeObserver) {
                        this.resizeObserver.disconnect();
                        this.resizeObserver = undefined;
                    }
                    sessionStorage.removeItem('contactFormEmail');
                    sessionStorage.removeItem('contactFormMessage');
                    this.modalContent.innerHTML = `
                        <div class="form-success-message">
                            <p>Thanks for your message!<br>I'll get back to you as soon as possible.</p>
                            <button id="ok-button">OK</button>
                        </div>`;
                } else {
                    const responseData = await response.json();
                    status.innerHTML = responseData.errors?.map((error: { message: string }) => error.message).join(", ") || "Oops! There was a problem submitting your form.";
                    status.className = 'form-status error';
                }
            } catch (error) {
                console.error('Form submission error:', error);
                status.innerHTML = "Oops! There was a problem submitting your form.";
                status.className = 'form-status error';
            }
        });
    }

    public static getInstance(): ModalDialog {
        if (!ModalDialog.instance) {
            ModalDialog.instance = new ModalDialog();
        }
        return ModalDialog.instance;
    }

    public show(content: string) {
        // If a resize observer is active from a previous view, disconnect it.
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = undefined;
        }

        // Only set content if it's not already there, to preserve form state
        if (this.modalContent.innerHTML !== content) {
            this.modalContent.innerHTML = content;
        }
        this.overlay.style.display = 'flex';

        // Push a state to the history so the back button can be used to close the modal.
        if (!history.state?.modalOpen) {
            history.pushState({ modalOpen: true }, '', window.location.href);
        }

        // Restore form state
        const emailInput = this.modalContent.querySelector('#email') as HTMLInputElement;
        const messageInput = this.modalContent.querySelector('#message') as HTMLTextAreaElement;

        if (emailInput && messageInput) {
            emailInput.value = sessionStorage.getItem('contactFormEmail') || '';
            messageInput.value = sessionStorage.getItem('contactFormMessage') || '';

            // When the textare is resized by the user, its container (.modal-content)
            // should naturally resize with it. A ResizeObserver ensures that
            // the browser recalculates layout when the textarea size changes,
            // making the dialog adjust its size dynamically.
            this.resizeObserver = new ResizeObserver(() => {
                // The presence of the observer is often enough to trigger a reflow.
            });
            this.resizeObserver.observe(messageInput);
        }
    }

    public hide(fromPopState = false) {
        if (this.overlay.style.display === 'none') {
            return;
        }

        // If not triggered by popstate, go back in history.
        if (!fromPopState && history.state?.modalOpen) {
            history.back();
        }

        // Save form state
        const emailInput = this.modalContent.querySelector('#email') as HTMLInputElement;
        const messageInput = this.modalContent.querySelector('#message') as HTMLTextAreaElement;

        if (emailInput && messageInput) {
            sessionStorage.setItem('contactFormEmail', emailInput.value);
            sessionStorage.setItem('contactFormMessage', messageInput.value);
        }

        // Disconnect the observer when hiding to clean up resources.
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = undefined;
        }

        this.overlay.style.display = 'none';
    }
}
