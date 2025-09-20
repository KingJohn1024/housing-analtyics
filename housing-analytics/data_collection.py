import os
import json
import time
import signal
import logging
import requests
import numpy as np
import pandas as pd
import undetected_chromedriver as uc
from datetime import date
from lxml.html import fromstring  # Replace this call?
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class DataCollection(object):
    def __init__(self, verbose=False):
        # zillow_api_key = '9202cb6d35msh267f818513d4c85p1364fdjsn3cbef17f0867'
        # zillow_api_key = '77b988d7b0mshe29695b4b1b70e4p143e3cjsnf8aebb0e1e9d'
        # zillow_api_key = 'fcceabeb9amshbf564b56f3106afp1ed137jsn86bb664919c2'
        # zillow_api_key = '322d8225bfmsh27bf206ed5a9ac1p16fceejsn20980c1afc0b'
        # b183b26252msh3407ce7bff0814fp117a74jsn890671917ddc
        # 7044be31eemsh278b3d382abfd55p1a1fbcjsn24044b859476
        # e2168eb707msh4b307843c2da548p17279ejsn291624712ef8
        # 91f66d2f09msh8dc35b159089680p104960jsn5d64660af0fb
        # 99b0909d72mshbd9ab77a45a3866p11e957jsn1a39b88f6308
        # d9725d8ae0msh9cc08f8d952a3c4p1ef844jsn4f55ced906d0
        # ea0c6df2d1mshccd1bb367b4a2bap1b7ecbjsn44477cc7dbed
        # d85572692bmshafabd3e3490a627p199f7djsn413da74bd549
        self.zillow_api_key = 'd85572692bmshafabd3e3490a627p199f7djsn413da74bd549'

        self.url = 'https://zillow56.p.rapidapi.com'
        self.search_url = f'{self.url}/search'
        self.headers = {
            'X-RapidAPI-Key': self.zillow_api_key,
            'X-RapidAPI-Host': 'zillow56.p.rapidapi.com',
        }
        self.rate_limit_error = {
            'message': 'You have exceeded the rate limit per second for your plan, BASIC, by the API provider'
        }

        # Set the logging level based on the verbose parameter
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.CRITICAL)

    def get_proxy_server(self, require_ssl=False):
        """
        Get proxy server IP address

        :param require_ssl: Boolean flag to indicate if SSL functionality is required
        :return: IP address of the proxy server
        """
        url = 'https://free-proxy-list.net/'
        response = requests.get(url, timeout=5.0)
        parser = fromstring(response.text)
        proxies = set()
        for i in parser.xpath('//tbody/tr')[:299]:  # 299 proxies max
            https = i.xpath('.//td[7]/text()')[0]
            if require_ssl:
                if https == 'no':
                    continue
            proxy = ':'.join(
                [i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]]
            )
            proxies.add(proxy)
        return np.random.choice(list(proxies))

    def get_listings(
        self,
        location: str = 'south boston, boston, ma',
        status: str = 'recentlySold',
    ):
        """
        Get listing details for a given location / status from Zillow API

        :param location: search location string
        :returns: Pandas dataframe of results
        """
        query_string = {
            'location': location,
            'page': 1,
        }

        if status:
            query_string['status'] = status
        # querystring = {"location":"02127", 'page': 1, 'status': 'recentlySold'}
        page = 1

        response = requests.get(
            self.search_url, headers=self.headers, params=query_string
        ).json()
        data = response.get('results')
        total_pages = response.get('totalPages', 1)

        while total_pages > page:
            page += 1
            self.logger.info(f'Requesting additional pages: {page}')
            query_string['page'] = page
            response = requests.get(
                self.search_url, headers=self.headers, params=query_string
            ).json()
            self.logger.info(
                f"Number of results per-page: {len(response.get('results', []))}"
            )
            data += response.get('results', [])

            # The API is silently rate-limited, sleep 10 seconds in between resquests to avoid collecting empty batches
            time.sleep(10)

        df = pd.DataFrame(data).rename(columns={'dateSold': 'SoldDate'})
        if 'list_sub_type' in df.columns:
            df.drop(columns=['list_sub_type'], inplace=True)
        # SoldDate only applys to sales!
        if 'SoldDate' in df.columns:
            df['SoldDate'] = pd.to_datetime(df.SoldDate, unit='ms').dt.date
        df['datePriceChanged'] = pd.to_datetime(df.datePriceChanged, unit='ms').dt.date
        df['ListingDate'] = np.NaN  # init null ListingDate value | sourced seperately
        df['url'] = df.zpid.apply(
            lambda xs: f'http://www.zillow.com/homedetails/{xs}_zpid/'
        )
        return df

    def upsert_frame(self, df: pd.DataFrame, frame_name: str, key: str):
        fpath = f'./data/{frame_name}.pq'
        if os.path.exists(fpath):
            existing = pd.read_parquet(fpath)
            combined = pd.concat(
                [
                    existing.dropna(axis=1, how='all'),
                    df[~df[key].isin(existing[key])].dropna(axis=1, how='all'),
                ]
            )
            combined.to_parquet(fpath)
            df = combined
        else:
            df.to_parquet(fpath)
        return df

    def get_property_profile(self, zpid):
        """
        Get property profile details for a given zpid and save locally as json file
        Example vicksburg_zpid = 448689951

        :param zpid: Zillow property identifier
        :return: None
        """
        print(f'{self.url}/propertyV2?zpid={zpid}')
        data = requests.get(
            f'{self.url}/propertyV2?zpid={zpid}',
            headers=self.headers,
        ).json()
        if data == self.rate_limit_error:
            # raise Exception('Rate limit hit, json return is warning only')
            self.logger('Hit rate limit, sleeping for 10 seconds')
            time.sleep(10)
        fpath = f'./data/property_profiles/{zpid}.json'
        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_zillow_listing_date_selenium(
        self, zpid: str, driver: uc.Chrome = None, headless: bool = True
    ):
        """
        Get the Zillow listing date for a given zpid by loading the home details page and scraping the HTML. The
        listing date values from the API are often missing or inconsistent. This is a critical data element
        for the accurately modeling DSO so extra effort has been invested to ensure the accuracy of this data. Web
        automation / scraping is very fragile, the logic here was developing using render page inspection and trail /
        error. This is the best coverage I could come up with and adequate for the model task at hand.

        :param zpid: Zillow property identifier
        :param driver: Chrome driver for web scraping automation (shared external driver for more control)
        :param headless: Boolean flag indicating if the Chrome automation sessions should run headless or not
        :return: Listing date
        """
        shared_driver = isinstance(driver, uc.Chrome)

        if not isinstance(driver, uc.Chrome):
            options = uc.ChromeOptions()
            options.headless = headless
            driver = uc.Chrome(
                user_data_dir='profile', options=options
            )  # User data dir forces a shared parent UC session
            driver.implicitly_wait(2)

        url = f'http://www.zillow.com/homedetails/{zpid}_zpid/'
        listing_date = np.NaN

        try:
            self.logger.info(url)
            driver.get(url)
            time.sleep(15)
            self.logger.info(f'Page Title: {driver.title}')
        except Exception as e:
            self.logger.info(f'URL Get Exception: {e}')

        # Try to find and click "Show more" if it exists
        try:
            show_more_button = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//button[span[normalize-space()='Show more']][preceding::td[span[@data-testid='date-info']]]",
                    )
                )
            )
            # Click via JavaScript to avoid interception issues
            driver.execute_script('arguments[0].click();', show_more_button)
            self.logger.info("Clicked 'Show more' button.")
        except:
            self.logger.info("No 'Show more' button found. Proceeding...")

        try:
            show_more_button = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//button[span[normalize-space()='Show more']][following::td[span[@data-testid='date-info']]]",
                    )
                )
            )
            driver.execute_script('arguments[0].click();', show_more_button)
            self.logger.info("Clicked 'Show more' button 2.")
        except:
            self.logger.info("No 'Show more' button 2 found. Proceeding...")

        try:
            listing_date_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//td[span[@data-testid='date-info']]/following-sibling::td[1]/span[normalize-space()='Listed for sale']/parent::td/preceding-sibling::td[1]/span",
                    )
                )
            )
            if not listing_date_element:
                self.logger.info('Trying to find Listing Date round 2')
                listing_date_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//td[span[normalize-space()='Listed for sale']]/preceding-sibling::td[1]/span[@data-testid='date-info']",
                        )
                    )
                )

            listing_date = (
                listing_date_element.text if listing_date_element else 'Not found'
            )
        except Exception as e:
            self.logger.info(f'Round 1 Error: {e}')
            try:
                self.logger.info('Trying to find Listing Date round 2')
                listing_date_element = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//td[span[normalize-space()='Listed for sale']]/preceding-sibling::td[1]/span[@data-testid='date-info']",
                        )
                    )
                )

                listing_date = (
                    listing_date_element.text if listing_date_element else 'Not found'
                )
            except Exception as e:
                self.logger.info(f'Round 2 Error: {e}')
                self.logger.info('Listing date not found.')
        finally:
            if not shared_driver:
                driver.close()  # Close the current window
                driver.quit()  # Shuts down the entire driver process

        if not shared_driver:
            # Clean up Chrome driver instances
            driver.close()
            driver.quit()
            os.kill(driver.service.process.pid, signal.SIGTERM)

        self.logger.info(f'Found Lisiting Date: {listing_date}')
        return listing_date

    def append_listing_date(
        self,
        df_sold: pd.DataFrame,
        headless: bool = True,
        use_proxy_server: bool = False,
    ):
        """
        Append the listing date from the Zillow home details webpage
        :param df_sold: Pandas Dataframe containing Zillow listing details
        :param headless: Boolean flag indicating if the Chrome automation sessions should run headless or not
        :param use_proxy_server: Boolean flag inidcating the use of random proxy server when access Zillow
        """
        coverage = df_sold[df_sold.ListingDate.notnull()].shape[0] / df_sold.shape[0]
        self.logger.info(f'Current Listing Date Coverage: {coverage:.2%}')

        # Initial a single Chrome driver to be shared
        options = uc.ChromeOptions()
        options.headless = headless
        if use_proxy_server:
            proxy = self.get_proxy_server(require_ssl=False)
            options.add_argument(f'--proxy-server=http://{proxy}')
        # User data dir forces a shared parent UC session
        shared_driver = uc.Chrome(user_data_dir='profile', options=options)
        shared_driver.implicitly_wait(2)

        # Force the datatype to avoid warnings when setting a data to an existing float (NaN) column
        df_sold['ListingDate'] = pd.to_datetime(df_sold['ListingDate'], errors='coerce')
        for zpid in df_sold[df_sold.ListingDate.isnull()].zpid.values:
            listing_date = self.get_zillow_listing_date_selenium(
                zpid, driver=shared_driver, headless=headless
            )
            self.logger.info(f'ZPID: {zpid} - Listing Date: {listing_date}')
            df_sold.loc[df_sold.zpid == int(zpid), 'ListingDate'] = listing_date
            df_sold.to_parquet('./data/df_sold.pq')  # Refresh the data

        # The driver could be in a failed state, in which case calling close will throw an exception
        try:
            shared_driver.close()
        except Exception as _:
            pass
        shared_driver.quit()
        os.kill(shared_driver.service.process.pid, signal.SIGTERM)

    def load_property_profile(self, zpid):
        fpath = f'./data/property_profiles/{zpid}.json'
        if not os.path.exists(fpath):
            self.get_property_profile(zpid=zpid)
        with open(fpath, 'r') as f:
            pp = json.load(f)
        return pp

    def run_data_collection(self):
        # zero - init the storage location
        # Master params
        location = 'south boston, boston, ma'

        ##########################################################
        # Step 1 - Collection all recently sold listing information
        df_sold = self.get_listings(location=location, status='recentlySold')
        df_sold = self.upsert_frame(df=df_sold, frame_name='df_sold', key='zpid')

        ##########################################################
        # Step 2 - Collection open listings for the given location
        df_listing = self.get_listings(location=location, status=None)
        df_listing = self.upsert_frame(
            df=df_listing, frame_name='df_listing', key='zpid'
        )

        ##########################################################
        # Step 3 - Collect property profile data for all zpids
        all_zpids = list(set(df_sold.zpid.to_list() + df_listing.zpid.to_list()))
        existing_zpids = [
            int(f.split('.')[0]) for f in os.listdir('./data/property_profiles')
        ]
        process_zpids = list(set(all_zpids) - set(existing_zpids))
        for zpid in process_zpids:
            self.get_property_profile(zpid=zpid)

        ##########################################################
        # Step 4 - Fetch the actual "Listing Date" and append to the data
        # Running with headless=True can trigger bot detection, run with headless = False using a shared session
        self.append_listing_date(
            df_sold=df_sold, headless=False, use_proxy_server=False
        )


if __name__ == '__main__':
    dc = DataCollection(verbose=True)
    dc.run_data_collection()
