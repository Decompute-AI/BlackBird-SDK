"""
Authentication routes module for Decompute Windows backend
Handles Google OAuth authentication, token verification, and user management
"""

from flask import Blueprint, request, jsonify, redirect
import pymysql
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# Import from core modules
from core.config import (
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, 
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    ALLOWED_COUNTRIES
)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

def get_db_connection():
    """Create database connection using configuration"""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

@auth_bp.route('/login', methods=['POST'])
def login():
    """Handle user login with Google OAuth and location verification"""
    data = request.get_json()
    token = data.get('token')               # Google ID token
    location_data = data.get('location')      # Expected to contain { lat, lng, place }
    
    if not token:
        return jsonify({'error': 'No token provided'}), 400

    try:
        # Verify the token using Google's library
        id_info = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        google_id      = id_info.get('sub')
        email          = id_info.get('email')
        email_verified = id_info.get('email_verified', False)
        name           = id_info.get('name')
        given_name     = id_info.get('given_name')
        family_name    = id_info.get('family_name')
        picture        = id_info.get('picture')
        locale         = id_info.get('locale')
    except ValueError as e:
        # Token is invalid
        return jsonify({'error': 'Invalid token', 'details': str(e)}), 400

    # Extract country from the provided country data
    country = None
    if location_data and location_data.get('place'):
        # Reverse geocoded place is expected to be an object with keys like city, state, country
        country = location_data['place'].get('country')
    
    if not country:
        return jsonify({'error': 'country data missing country information.'}), 400

    # Restrict access to only allowed countries
    if country not in ALLOWED_COUNTRIES:
        return jsonify({'error': 'Access restricted to users from India and USA only.'}), 403

    # Upsert user into the database with the country information
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        upsert_query = """
            INSERT INTO users 
                (google_id, email, email_verified, name, given_name, family_name, picture, locale, country)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                email = VALUES(email),
                email_verified = VALUES(email_verified),
                name = VALUES(name),
                given_name = VALUES(given_name),
                family_name = VALUES(family_name),
                picture = VALUES(picture),
                locale = VALUES(locale),
                country = VALUES(country);
        """
        cursor.execute(upsert_query, (
            google_id, email, email_verified, name, given_name,
            family_name, picture, locale, country
        ))
        conn.commit()

        # Retrieve the updated user record
        select_query = "SELECT * FROM users WHERE google_id = %s"
        cursor.execute(select_query, (google_id,))
        user = cursor.fetchone()

        cursor.close()
        conn.close()
        return jsonify({'user': user, 'detected_country': country})
    except Exception as e:
        return jsonify({'error': 'Database error', 'details': str(e)}), 500

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    """Verify Google OAuth token"""
    try:
        # Get token from request body
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'valid': False, 'error': 'No token provided'}), 400

        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        # Token is valid
        return jsonify({'valid': True})

    except ValueError as e:
        # Invalid token
        print(f"Token verification failed: {str(e)}")
        return jsonify({'valid': False})

    except Exception as e:
        # Other errors
        print(f"Error verifying token: {str(e)}")
        return jsonify({'valid': False})


@auth_bp.route('/oauth/callback')
def oauth_callback():
    try:
        # 1. Get the authorization code from the callback
        code = request.args.get('code')
        
        if not code:
            return jsonify({'error': 'No authorization code received'}), 400
        
        # 2. Exchange the code for tokens
        token_url = 'https://oauth2.googleapis.com/token'
        data = {
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'code': code,
            'redirect_uri': 'http://127.0.0.1:5012/oauth/callback',  # Must match Google OAuth config
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(token_url, data=data)
        
        if not response.ok:
            return jsonify({'error': f'Token exchange failed: {response.text}'}), response.status_code
            
        tokens = response.json()
        
        # 3. Verify the ID token and gather basic user info
        id_info = id_token.verify_oauth2_token(
            tokens['id_token'], 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        user_payload = {
            'google_id': id_info.get('sub'),
            'email': id_info.get('email'),
            'name': id_info.get('name'),
            'picture': id_info.get('picture')
        }
        
        # 4. Create a simple HTML page with a button
        #
        #    Instead of automatically redirecting, we show a button
        #    that the user can click to open your app via the custom
        #    protocol ("decompute://..."). That click is considered
        #    a valid user action, so Chrome/Edge won't block it.
        
        html_response = f"""
        <html>
        <head>
            <title>Authentication Successful</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    text-align: center;
                    padding: 50px;
                    background-color: #f0f4f8;
                    color: #333;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    background-color: #fff;
                }}
                h2 {{
                    color: #1c7df2;
                    font-size: 2.5em;
                    margin-bottom: 20px;
                }}
                p {{
                    font-size: 1.2em;
                    margin: 10px 0;
                }}
                .button {{
                    display: inline-block;
                    margin-top: 30px;
                    padding: 15px 30px;
                    font-size: 1em;
                    color: #fff;
                    background-color: #1c7df2;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                }}
                .button:hover {{
                    background-color: #0b5ed7;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Authentication Successful!</h2>
                <p>Welcome, {user_payload['name']}!</p>
                <p>Your email: {user_payload['email']}</p>
                
                <p style="margin-top: 20px;">
                  Click the button below to return to the BlackBird application.
                </p>
                <a 
                    class="button"
                    href="decompute://auth?token={tokens['id_token']}"
                >
                  Open BlackBird
                </a>

                <p style="margin-top: 40px; font-size: 0.9em; color: #666;">
                    If nothing happens, please open the BlackBird app manually and sign in again.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html_response
        
    except Exception as e:
        return jsonify({'error': f'OAuth error: {str(e)}'}), 500

