"""
Payment routes module for Decompute Windows backend
Handles Stripe payment processing, subscriptions, and billing management
"""

from flask import Blueprint, request, jsonify, redirect, render_template
import stripe
import requests
import json
import logging
from datetime import datetime

# Import from core modules
from core.config import (
    STRIPE_API_KEY, YOUR_DOMAIN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
)

# Create blueprint
payment_bp = Blueprint('payment', __name__)

# Initialize Stripe
stripe.api_key = STRIPE_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

def find_subscription_id(email):
    """Find subscription ID for a given email address"""
    try:
        # First API call: Get customer by email
        customers_response = requests.get(
            'https://api.stripe.com/v1/customers',
            params={
                'limit': 3,
                'email': email  # Filter by email
            },
            auth=(STRIPE_API_KEY, '')
        )
        customers_data = customers_response.json()
        logger.info(f"Customers response: {customers_data}")

        # Check if we found the customer
        if not customers_data.get('data'):
            logger.info(f"No customer found for email: {email}")
            return None

        customer_id = customers_data['data'][0]['id']
        logger.info(f"Found customer ID: {customer_id}")

        # Second API call: Get subscriptions for this customer
        subscriptions_response = requests.get(
            'https://api.stripe.com/v1/subscriptions',
            params={
                'limit': 3,
                'customer': customer_id  # Filter by customer
            },
            auth=(STRIPE_API_KEY, '')
        )
        subscriptions_data = subscriptions_response.json()
        logger.info(f"Subscriptions response: {subscriptions_data}")

        # Find subscription with our price ID
        pro_price_id = 'price_1R02agFbGwqVxzy7ghhj1NAr'
        subscription_id = None

        for subscription in subscriptions_data.get('data', []):
            for item in subscription['items']['data']:
                if item['price']['id'] == pro_price_id:
                    subscription_id = subscription['id']
                    break
            if subscription_id:
                break

        if subscription_id:
            # Third API call: Get specific subscription details
            subscription_response = requests.get(
                f'https://api.stripe.com/v1/subscriptions/{subscription_id}',
                auth=(STRIPE_API_KEY, '')
            )
            subscription_details = subscription_response.json()
            logger.info(f"Subscription details: {subscription_details}")
            
            return subscription_id
        
        logger.info(f"No pro subscription found for customer: {customer_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error finding subscription ID: {str(e)}")
        return None

@payment_bp.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    """Create a Stripe checkout session for subscription"""
    try:
        data = request.get_json()
        lookup_key = data.get('lookup_key')
        email = data.get('email')
        
        # Create a new checkout session
        checkout_session = stripe.checkout.Session.create(
            billing_address_collection='auto',
            line_items=[
                {
                    'price': 'price_1R02agFbGwqVxzy7ghhj1NAr',
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=request.headers.get('Origin', YOUR_DOMAIN) + 
                        '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url= 'http://127.0.0.1:5012/canceled',
            customer_email=email,  # Pre-fill the email field
        )
        
        # Return the checkout session URL
        return jsonify({'url': checkout_session.url})
    
    except Exception as e:
        logger.error(f'Error creating checkout session: {str(e)}')
        return jsonify({'error': str(e)}), 500

@payment_bp.route('/create-portal-session', methods=['POST'])
def customer_portal():
    """Create a customer portal session for subscription management"""
    try:
        # For demonstration purposes, we're using the Checkout session to retrieve the customer ID.
        # Typically this is stored alongside the authenticated user in your database.
        checkout_session_id = request.form.get('session_id')
        checkout_session = stripe.checkout.Session.retrieve(checkout_session_id)

        # This is the URL to which the customer will be redirected after they're
        # done managing their billing with the portal.
        return_url = YOUR_DOMAIN

        portalSession = stripe.billing_portal.Session.create(
            customer=checkout_session.customer,
            return_url=return_url,
        )
        return redirect(portalSession.url, code=303)
    
    except Exception as e:
        logger.error(f'Error creating portal session: {str(e)}')
        return jsonify({'error': str(e)}), 500

@payment_bp.route('/check-subscription', methods=['POST'])
def check_subscription():
    """Check subscription status for a given email"""
    try:
        logger.info("Starting subscription check")
        data = request.json
        email = data.get('email')
        
        if not email:
            logger.warning("No email provided in request")
            return jsonify({"error": "Email is required"}), 400

        try:
            # Find subscription ID using the sequence of API calls
            subscription_id = find_subscription_id(email)  # Pass email to the function
            
            if not subscription_id:
                return jsonify({
                    "isPro": False,
                    "message": "No subscription found"
                })

            # Get subscription details
            subscription_response = requests.get(
                f'https://api.stripe.com/v1/subscriptions/{subscription_id}',
                auth=(STRIPE_API_KEY, '')
            )
            subscription = subscription_response.json()

            # Get customer details
            customer_response = requests.get(
                f'https://api.stripe.com/v1/customers/{subscription["customer"]}',
                auth=(STRIPE_API_KEY, '')
            )
            customer = customer_response.json()

            return jsonify({
                "isPro": True,
                "status": subscription['status'],
                "currentPeriodStart": subscription['current_period_start'],
                "currentPeriodEnd": subscription['current_period_end'],
                "interval": subscription['items']['data'][0]['price']['recurring']['interval'],
                "cancelAtPeriodEnd": subscription['cancel_at_period_end'],
                "cancelAt": subscription.get('cancel_at'),
                "endedAt": subscription.get('ended_at'),
                "subscriptionId": subscription['id'],
                "customer": {
                    "email": customer['email'],
                    "name": customer.get('name')
                }
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            return jsonify({"error": f"API request error: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error in check_subscription: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to check subscription status",
            "details": str(e)
        }), 500

@payment_bp.route('/cancel-subscription', methods=['POST'])
def cancel_subscription():
    """Cancel a user's subscription"""
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Find customer by email
        customers = stripe.Customer.list(email=email, limit=1).data
        
        if not customers:
            return jsonify({"error": "No customer found with this email"}), 404
            
        customer_id = customers[0].id
        
        # Get active subscriptions for this customer
        subscriptions = stripe.Subscription.list(
            customer=customer_id,
            status='active',
            limit=100
        ).data
        
        # Find the Pro subscription
        pro_price_id = 'price_1R02agFbGwqVxzy7ghhj1NAr'  # Your Pro plan price ID
        pro_subscription = None
        
        for subscription in subscriptions:
            for item in subscription.items.data:
                if item.price.id == pro_price_id:
                    pro_subscription = subscription
                    break
            if pro_subscription:
                break
                
        if not pro_subscription:
            return jsonify({"error": "No active Pro subscription found"}), 404
            
        # Cancel the subscription at the end of the current period
        canceled_subscription = stripe.Subscription.modify(
            pro_subscription.id,
            cancel_at_period_end=True
        )
        
        return jsonify({
            "success": True,
            "message": "Subscription will be canceled at the end of the billing period",
            "subscriptionId": canceled_subscription.id,
            "cancelDate": canceled_subscription.cancel_at
        })
        
    except Exception as e:
        return jsonify({
            "error": "Failed to cancel subscription",
            "details": str(e)
        }), 500

@payment_bp.route('/webhook', methods=['POST'])
def webhook_received():
    """Handle Stripe webhook events"""
    webhook_secret = 'whsec_12345'
    request_data = json.loads(request.data)

    if webhook_secret:
        # Retrieve the event by verifying the signature using the raw body and secret if webhook signing is configured.
        signature = request.headers.get('stripe-signature')
        try:
            event = stripe.Webhook.construct_event(
                payload=request.data, sig_header=signature, secret=webhook_secret)
            data = event['data']
        except Exception as e:
            return e
        # Get the type of webhook event sent - used to check the status of PaymentIntents.
        event_type = event['type']
    else:
        data = request_data['data']
        event_type = request_data['type']
    data_object = data['object']

    logger.info('event ' + event_type)

    if event_type == 'checkout.session.completed':
        logger.info('🔔 Payment succeeded!')
    elif event_type == 'customer.subscription.trial_will_end':
        logger.info('Subscription trial will end')
    elif event_type == 'customer.subscription.created':
        logger.info('Subscription created %s', event.id)
    elif event_type == 'customer.subscription.updated':
        logger.info('Subscription created %s', event.id)
    elif event_type == 'customer.subscription.deleted':
        # handle subscription canceled automatically based
        # upon your subscription settings. Or if the user cancels it.
        logger.info('Subscription canceled: %s', event.id)
    elif event_type == 'entitlements.active_entitlement_summary.updated':
        # handle active entitlement summary updated
        logger.info('Active entitlement summary updated: %s', event.id)

    return jsonify({'status': 'success'})

@payment_bp.route('/verify-payment', methods=['POST'])
def verify_payment():
    """Verify payment completion"""
    try:
        session_id = request.json.get('sessionId')
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Verify payment was successful
        if session.payment_status == 'paid':
            # Update your database, etc.
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'reason': 'Payment not completed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@payment_bp.route('/success')
def success():
    """Payment success page"""
    session_id = request.args.get('session_id')
    
    # Verify the session payment was successful
    session = stripe.checkout.Session.retrieve(session_id)
    
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Successful</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            @keyframes pulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.05); }
            }
            .animate-pulse-slow {
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            .success-checkmark {
                width: 80px;
                height: 80px;
                margin: 0 auto;
                border-radius: 50%;
                display: block;
                stroke-width: 2;
                stroke: #4bb71b;
                stroke-miterlimit: 10;
                box-shadow: 0px 0px 20px rgba(75, 183, 27, 0.3);
            }
            .success-checkmark .check-icon {
                width: 80px;
                height: 80px;
                stroke-width: 2;
                stroke: #FFFFFF;
                stroke-miterlimit: 10;
                transform: translate(10%, 10%);
            }
            .progress-bar {
                width: 100%;
                height: 5px;
                background-color: #e2e8f0;
                border-radius: 5px;
                overflow: hidden;
                margin: 20px 0;
            }
            .progress-bar-value {
                width: 0%;
                height: 100%;
                background-color: #4bb71b;
                border-radius: 5px;
                animation: progress 5s linear forwards;
            }
            @keyframes progress {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
                height: 100vh;
            }
        </style>
    </head>
    <body class="flex items-center justify-center h-screen">
        <div class="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
            <div class="text-center">
                <div class="success-checkmark animate-pulse-slow mb-6">
                    <svg viewBox="0 0 52 52" class="check-icon">
                        <circle cx="26" cy="26" r="25" fill="#4bb71b" />
                        <path fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8" stroke-width="4" stroke="white" stroke-linecap="round" />
                    </svg>
                </div>
                
                <h1 class="text-2xl font-bold text-gray-800 mb-2">Payment Successful!</h1>
                <p class="text-gray-600 mb-6">Thank you for your subscription.</p>
                
                <div class="progress-bar">
                    <div class="progress-bar-value"></div>
                </div>
                
                <p class="text-gray-600 mb-6">Redirecting you back to the application...</p>
                
                <div id="manual-return" class="hidden mt-6">
                    <p class="text-sm text-gray-500 mb-4">If the app doesn't open automatically, close this window :</p>
                </div>
            </div>
        </div>
        
        <script>
            // Try to redirect to your Electron app via custom protocol
            function triggerAppOpen() {
                window.location.href = 'yourapp://payment-success?session_id=""" + session_id + """';
            }
            
            // Attempt automatic redirect
            setTimeout(triggerAppOpen, 2000);
            
            // Fallback: If after 5 seconds we're still here, show manual button
            setTimeout(function() {
                document.getElementById('manual-return').classList.remove('hidden');
            }, 5000);
        </script>
    </body>
    </html>
    """

@payment_bp.route('/canceled')
def canceled():
    """Payment canceled page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Canceled</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            .progress-bar {
                width: 100%;
                height: 5px;
                background-color: #e2e8f0;
                border-radius: 5px;
                overflow: hidden;
                margin: 20px 0;
            }
            .progress-bar-value {
                width: 0%;
                height: 100%;
                background-color: #3b82f6;
                border-radius: 5px;
                animation: progress 5s linear forwards;
            }
            @keyframes progress {
                0% { width: 0%; }
                100% { width: 100%; }
            }
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
                height: 100vh;
            }
            .notice-icon {
                width: 80px;
                height: 80px;
                margin: 0 auto;
                display: block;
                stroke-width: 2;
                border-radius: 50%;
                background-color: #f3f4f6;
                box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body class="flex items-center justify-center h-screen">
        <div class="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
            <div class="text-center">
                <div class="notice-icon mb-6 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="12" y1="8" x2="12" y2="12" />
                        <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                </div>
                
                <h1 class="text-2xl font-bold text-gray-800 mb-2">Payment Canceled</h1>
                <p class="text-gray-600 mb-6">Your payment was not completed.</p>
                
                <div class="progress-bar">
                    <div class="progress-bar-value"></div>
                </div>
                
                <p class="text-gray-600 mb-6">Redirecting you back to the application...</p>
                
                <div id="manual-return" class="hidden mt-6">
                    <p class="text-sm text-gray-500 mb-4">If the app doesn't open automatically, close this window:</p>
                </div>
            </div>
        </div>
        
        <script>
            // Try to redirect to your Electron app via custom protocol
            function triggerAppOpen() {
                window.location.href = 'yourapp://payment-canceled';
            }
            
            // Attempt automatic redirect
            setTimeout(triggerAppOpen, 2000);
            
            // Fallback: If after 5 seconds we're still here, show manual button
            setTimeout(function() {
                document.getElementById('manual-return').classList.remove('hidden');
            }, 5000);
        </script>
    </body>
    </html>
    """
