import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, receiver_email, subject, message):
    # Set up the SMTP server
    smtp_server = "smtp.gmail.com"  # For Gmail
    smtp_port = 587  # For Gmail

    # Create a MIMEText object to represent the email message
    email_message = MIMEMultipart()
    email_message['From'] = sender_email
    email_message['To'] = receiver_email
    email_message['Subject'] = subject
    email_message.attach(MIMEText(message, 'plain'))

    # Connect to the SMTP server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Start TLS encryption
    server.login(sender_email, sender_password)

    # Send the email
    server.sendmail(sender_email, receiver_email, email_message.as_string())

    # Close the connection
    server.quit()

# Example usage:
sender_email = "adityapatildev2810@gmail.com"
sender_password = "iopw ecxc jrgi fubf"
receiver_email = "adityapatilsy@gmail.com"
subject = "Test Email"
message = "This is a test email sent from Python."

send_email(sender_email, sender_password, receiver_email, subject, message)
