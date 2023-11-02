import boto3
from boto3.s3.transfer import S3Transfer
from botocore.exceptions import ClientError
from app.core.configs import get_environment, get_logger

_env = get_environment()
_logger = get_logger(__name__)


class Bucket:
    @staticmethod
    def __connect_on_client():
        bucket = boto3.client(
            "s3",
            endpoint_url=_env.BUCKET_BASE_URL,
            aws_access_key_id=_env.BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=_env.BUCKET_SECRET_KEY,
        )
        return bucket
    
    @staticmethod
    def __connect_on_resource():
        bucket = boto3.resource(
            "s3",
            endpoint_url=_env.BUCKET_BASE_URL,
            aws_access_key_id=_env.BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=_env.BUCKET_SECRET_KEY,
        )
        return bucket
    
    @classmethod
    def verify_bucket(cls):
        _logger.info(f"Checking Bucket")
        try:
            bucket = cls.__connect_on_resource()
            buckets = bucket.buckets.all()
            if buckets:
                buckets = [created_bucket.name for created_bucket in buckets]

            else:
                buckets = []

            if _env.BUCKET_NAME not in buckets:
                bucket.create_bucket(Bucket=_env.BUCKET_NAME)
                _logger.info(f"Bucket with name {_env.BUCKET_NAME} have been created")

            _logger.info(f"Bucket checked with success")

        except ClientError:
            _logger.error("Error on verify bucket")
            raise Exception("Bucket not created")

    @classmethod
    def save_file(cls, bucket_path: str, file_path: str) -> str:
        try:

            bucket = cls.__connect_on_client()
            transfer = S3Transfer(bucket)
            transfer.upload_file(file_path, _env.BUCKET_NAME, bucket_path)

            return f"{_env.BUCKET_BASE_URL}{_env.BUCKET_NAME}/{bucket_path}"

        except Exception as error:
            _logger.error(f"Error on save file in bucket: {str(error)}")
            raise Exception("Error on save file")

    @classmethod
    def get_presigned_url(cls, path: str) -> str:
        bucket = cls.__connect_on_client()
        url = bucket.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": _env.BUCKET_NAME, "Key": path},
            ExpiresIn=_env.BUCKET_URL_EXPIRES_IN_SECONDS,
        )
        return url
