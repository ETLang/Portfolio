export function getContactForm(): string {
  return `
    <h3>Contact Me</h3>
    <form 
      action="https://formspree.io/f/xdarwqjw"
      method="POST"
      class="contact-form"
    >
      <div class="form-group">
        <label for="email">Your Email</label>
        <input type="email" id="email" name="email" required>
      </div>
      <div class="form-group">
        <label for="message">Message</label>
        <textarea id="message" name="message" rows="5" required style="resize: vertical;"></textarea>
      </div>
      <button type="submit">Send</button>
      <p class="form-status"></p>
    </form>
  `;
}
